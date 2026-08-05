"""QLoRA fine-tuning with Unsloth (Phase 3).

Run on a CUDA GPU (T4/A100/RunPod). Unsloth and torch are imported lazily so
this module can be imported — and its config validated — on a machine with no
GPU at all.

    pip install -e ".[training]" && pip install unsloth
    python -m training.train_qlora \\
        --model unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit \\
        --epochs 2 --output training/adapters/omniserve-slm-8b

Hyper-parameters follow the build plan: r=16, alpha=32, batch 2 x grad-accum 4,
lr 2e-4, 10 warmup steps.
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from training.dataset import RESPONSE_MARKER, build_hf_dataset
from training.trl_compat import (
    filter_training_args,
    introspect,
    route_sft_options,
    tokenizer_kwarg,
)

TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


@dataclass
class TrainConfig:
    model: str = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    train_path: Path = Path("data/generated/train.jsonl")
    val_path: Path = Path("data/generated/val.jsonl")
    output: Path = Path("training/adapters/omniserve-slm-8b")
    max_seq_length: int = 4096
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    batch_size: int = 2
    grad_accum: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 10
    epochs: float = 2.0
    seed: int = 3407
    include_schema: bool = False
    completion_only: bool = True
    merge_16bit: bool = False
    target_modules: list[str] = field(default_factory=lambda: list(TARGET_MODULES))

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum


def build_config(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        model=args.model,
        train_path=args.train_path,
        val_path=args.val_path,
        output=args.output,
        max_seq_length=args.max_seq_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        epochs=args.epochs,
        seed=args.seed,
        include_schema=args.include_schema,
        completion_only=not args.train_on_prompt,
        merge_16bit=args.merge_16bit,
    )


def train(config: TrainConfig) -> None:
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from transformers import TrainingArguments
    from trl import SFTTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model,
        max_seq_length=config.max_seq_length,
        load_in_4bit=True,
        dtype=None,  # auto-detect
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.seed,
    )

    kwargs = {"include_schema": config.include_schema}
    train_dataset = build_hf_dataset(config.train_path, tokenizer.apply_chat_template, **kwargs)
    eval_dataset = build_hf_dataset(config.val_path, tokenizer.apply_chat_template, **kwargs)
    print(f"train={len(train_dataset)} val={len(eval_dataset)}")
    print(f"effective batch size = {config.effective_batch_size}")

    # SFTConfig is where modern TRL wants the SFT-specific options; older
    # versions take them on the trainer with plain TrainingArguments.
    try:
        from trl import SFTConfig as ArgsClass
    except ImportError:
        ArgsClass = TrainingArguments

    trainer_params, config_fields = introspect(SFTTrainer, ArgsClass)
    config_extras, trainer_extras = route_sft_options(
        trainer_params,
        config_fields,
        text_field="text",
        max_seq_length=config.max_seq_length,
        packing=False,  # packing would splice unrelated invoices into one window
    )

    requested_args = {
        "output_dir": str(config.output / "checkpoints"),
        "per_device_train_batch_size": config.batch_size,
        "gradient_accumulation_steps": config.grad_accum,
        "num_train_epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "warmup_steps": config.warmup_steps,
        "lr_scheduler_type": "linear",
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "fp16": not is_bfloat16_supported(),
        "bf16": is_bfloat16_supported(),
        "logging_steps": 25,
        "eval_strategy": "steps",
        "eval_steps": 250,
        "save_strategy": "steps",
        "save_steps": 500,
        "save_total_limit": 2,
        "seed": config.seed,
        "report_to": "none",
    }

    training_args = ArgsClass(
        **filter_training_args(requested_args, config_fields), **config_extras
    )

    tokenizer_key = tokenizer_kwarg(trainer_params)
    print(f"trl: args={ArgsClass.__name__} tokenizer_kwarg={tokenizer_key}")

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        **({tokenizer_key: tokenizer} if tokenizer_key else {}),
        **trainer_extras,
    )

    if config.completion_only:
        # Without this the model is scored on reproducing the invoice text it was
        # given, which is most of the sequence and none of the task.
        from unsloth.chat_templates import train_on_responses_only

        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|start_header_id|>user<|end_header_id|>",
            response_part=RESPONSE_MARKER,
        )

    trainer.train()

    config.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(config.output))
    tokenizer.save_pretrained(str(config.output))
    print(f"Saved LoRA adapter to {config.output}")

    if config.merge_16bit:
        merged = config.output.parent / f"{config.output.name}-merged"
        model.save_pretrained_merged(str(merged), tokenizer, save_method="merged_16bit")
        print(f"Saved merged 16-bit weights to {merged}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="train_qlora")
    parser.add_argument("--model", default=TrainConfig.model)
    parser.add_argument("--train-path", type=Path, default=TrainConfig.train_path)
    parser.add_argument("--val-path", type=Path, default=TrainConfig.val_path)
    parser.add_argument("--output", type=Path, default=TrainConfig.output)
    parser.add_argument("--max-seq-length", type=int, default=TrainConfig.max_seq_length)
    parser.add_argument("--lora-r", type=int, default=TrainConfig.lora_r)
    parser.add_argument("--lora-alpha", type=int, default=TrainConfig.lora_alpha)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--grad-accum", type=int, default=TrainConfig.grad_accum)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--warmup-steps", type=int, default=TrainConfig.warmup_steps)
    parser.add_argument("--epochs", type=float, default=TrainConfig.epochs)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument(
        "--include-schema",
        action="store_true",
        help="Keep the JSON schema in the prompt (~400 extra tokens of prefill per request).",
    )
    parser.add_argument(
        "--train-on-prompt",
        action="store_true",
        help="Score the whole sequence instead of the assistant turn only.",
    )
    parser.add_argument(
        "--merge-16bit",
        action="store_true",
        help="Also export merged 16-bit weights alongside the adapter.",
    )
    args = parser.parse_args()

    train(build_config(args))


if __name__ == "__main__":
    main()
