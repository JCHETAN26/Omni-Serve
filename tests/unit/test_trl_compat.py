"""Routing tests for TRL's moving signature.

Written against captured signature shapes rather than an installed TRL, so they
run in CI and still describe versions this machine has never had.
"""

from training.trl_compat import (
    filter_training_args,
    introspect,
    route_sft_options,
    tokenizer_kwarg,
)

# TRL <0.12: everything on the trainer, plain TrainingArguments.
LEGACY_TRAINER = {
    "self",
    "model",
    "args",
    "train_dataset",
    "eval_dataset",
    "tokenizer",
    "dataset_text_field",
    "max_seq_length",
    "packing",
}
LEGACY_CONFIG = {"output_dir", "learning_rate", "evaluation_strategy", "seed"}

# TRL >=0.12: SFT options moved into SFTConfig, tokenizer renamed.
MODERN_TRAINER = {"self", "model", "args", "train_dataset", "eval_dataset", "processing_class"}
MODERN_CONFIG = {
    "output_dir",
    "learning_rate",
    "eval_strategy",
    "seed",
    "dataset_text_field",
    "max_seq_length",
    "packing",
}

# TRL >=0.20: max_seq_length renamed to max_length.
NEWEST_CONFIG = (MODERN_CONFIG - {"max_seq_length"}) | {"max_length"}


def test_tokenizer_kwarg_prefers_the_modern_name():
    assert tokenizer_kwarg(MODERN_TRAINER) == "processing_class"
    assert tokenizer_kwarg(LEGACY_TRAINER) == "tokenizer"
    assert tokenizer_kwarg({"self", "model"}) is None


def test_legacy_routes_options_to_the_trainer():
    config_kwargs, trainer_kwargs = route_sft_options(
        LEGACY_TRAINER, LEGACY_CONFIG, text_field="text", max_seq_length=1024, packing=False
    )

    assert config_kwargs == {}
    assert trainer_kwargs == {
        "dataset_text_field": "text",
        "packing": False,
        "max_seq_length": 1024,
    }


def test_modern_routes_options_to_the_config():
    config_kwargs, trainer_kwargs = route_sft_options(
        MODERN_TRAINER, MODERN_CONFIG, text_field="text", max_seq_length=1024, packing=False
    )

    assert trainer_kwargs == {}
    assert config_kwargs["dataset_text_field"] == "text"
    assert config_kwargs["max_seq_length"] == 1024


def test_newest_binds_max_length_not_the_deprecated_alias():
    config_kwargs, _ = route_sft_options(
        MODERN_TRAINER, NEWEST_CONFIG, text_field="text", max_seq_length=1024, packing=False
    )

    assert config_kwargs["max_length"] == 1024
    assert "max_seq_length" not in config_kwargs


def test_unknown_option_is_dropped_rather_than_raising():
    """A dropped default is survivable; an unexpected kwarg kills the run."""
    config_kwargs, trainer_kwargs = route_sft_options(
        {"self", "model"}, {"output_dir"}, text_field="text", max_seq_length=1024, packing=False
    )

    assert config_kwargs == {}
    assert trainer_kwargs == {}


def test_eval_strategy_is_renamed_for_older_transformers():
    requested = {"output_dir": "out", "eval_strategy": "steps", "seed": 1}

    assert filter_training_args(requested, MODERN_CONFIG)["eval_strategy"] == "steps"
    assert filter_training_args(requested, LEGACY_CONFIG)["evaluation_strategy"] == "steps"


def test_unsupported_training_args_are_dropped():
    requested = {"output_dir": "out", "optim": "adamw_8bit", "bf16": True}

    accepted = filter_training_args(requested, {"output_dir"})

    assert accepted == {"output_dir": "out"}


def test_introspect_reads_dataclasses_and_callables():
    import dataclasses

    @dataclasses.dataclass
    class FakeConfig:
        output_dir: str = "x"
        seed: int = 0

    class FakeTrainer:
        def __init__(self, model, args, processing_class=None):
            pass

    trainer_params, config_fields = introspect(FakeTrainer, FakeConfig)

    assert "processing_class" in trainer_params
    assert config_fields == {"output_dir", "seed"}
