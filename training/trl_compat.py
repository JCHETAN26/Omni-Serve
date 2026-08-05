"""Version routing for TRL's moving SFTTrainer signature.

TRL and transformers have shuffled this API repeatedly, and Colab installs
whatever Unsloth pulls today — so pinning is a losing game. These helpers are
pure functions over the *installed* signatures, which keeps them unit-testable
without TRL present and lets training survive the next rename.

Known shapes:

    TRL <0.12   SFTTrainer(tokenizer=, dataset_text_field=, max_seq_length=,
                           packing=, args=TrainingArguments)
    TRL >=0.12  SFTTrainer(processing_class=, args=SFTConfig(dataset_text_field=,
                           max_seq_length=, packing=))
    TRL >=0.20  same, but SFTConfig calls it max_length

transformers >=4.46 also renamed `evaluation_strategy` to `eval_strategy`, and
removed Trainer's `tokenizer` argument outright — which is the TypeError this
module exists to prevent.
"""

# Ordered by preference: newest name first, so modern installs never bind a
# deprecated alias that still lingers in the signature.
SEQ_LEN_NAMES = ("max_length", "max_seq_length")
EVAL_STRATEGY_NAMES = ("eval_strategy", "evaluation_strategy")


def tokenizer_kwarg(trainer_params: set[str]) -> str | None:
    """Which keyword carries the tokenizer, if any."""
    for name in ("processing_class", "tokenizer"):
        if name in trainer_params:
            return name
    return None


def route_sft_options(
    trainer_params: set[str],
    config_fields: set[str],
    *,
    text_field: str,
    max_seq_length: int,
    packing: bool,
) -> tuple[dict, dict]:
    """Split SFT options between the config object and the trainer call.

    Returns (config_kwargs, trainer_kwargs). An option accepted by neither is
    dropped rather than raising: an unknown extra would fail the whole run,
    while a missing one usually just means the default is already correct.
    """
    config_kwargs: dict = {}
    trainer_kwargs: dict = {}

    def place(names: tuple[str, ...], value) -> None:
        for name in names:
            if name in config_fields:
                config_kwargs[name] = value
                return
        for name in names:
            if name in trainer_params:
                trainer_kwargs[name] = value
                return

    place(("dataset_text_field",), text_field)
    place(("packing",), packing)
    place(SEQ_LEN_NAMES, max_seq_length)
    return config_kwargs, trainer_kwargs


def filter_training_args(requested: dict, config_fields: set[str]) -> dict:
    """Drop or rename training arguments the installed version doesn't accept."""
    accepted: dict = {}
    for key, value in requested.items():
        if key in config_fields:
            accepted[key] = value
        elif key in EVAL_STRATEGY_NAMES:
            for alias in EVAL_STRATEGY_NAMES:
                if alias in config_fields:
                    accepted[alias] = value
                    break
    return accepted


def introspect(trainer_cls, config_cls) -> tuple[set[str], set[str]]:
    """Parameter names of the installed SFTTrainer and fields of its config."""
    import dataclasses
    import inspect

    trainer_params = set(inspect.signature(trainer_cls.__init__).parameters)

    if dataclasses.is_dataclass(config_cls):
        config_fields = {field.name for field in dataclasses.fields(config_cls)}
    else:
        config_fields = set(inspect.signature(config_cls.__init__).parameters)

    return trainer_params, config_fields
