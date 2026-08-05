"""Emit a lean training/colab_finetune.ipynb."""

import json
from pathlib import Path


def lines(text):
    """ipynb `source` is a list of lines that each KEEP their newline.

    Without the newlines the cell joins into a single line, which silently
    destroys shell line continuations and comments.
    """
    return text.strip().splitlines(keepends=True)


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": lines(text)}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(text),
    }


cells = [
    md("""
# OmniServe — QLoRA fine-tuning

Fine-tunes a domain SLM for invoice extraction and reports before/after accuracy.

**Runtime → Change runtime type → T4 GPU**, then Run all. ~90 minutes.
"""),
    md("## Setup"),
    code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"""),
    code("""
%%capture
# No TRL pin: Unsloth installs a compatible set, and training/trl_compat.py
# adapts to whichever SFTTrainer signature lands. Pinning here fought Unsloth
# and produced `Trainer.__init__() got an unexpected keyword argument 'tokenizer'`.
!pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install -q --no-deps peft accelerate bitsandbytes
"""),
    code("""
# Not captured: this is the cell most likely to fail, and a silent
# clone failure would surface much later as a confusing import error.
!git clone -q https://github.com/JCHETAN26/Omni-Serve.git omniserve
%cd omniserve
!pip install -q -e .
"""),
    md("""
## Config

Every knob lives here. The eval cells both read `LIMIT`, so the baseline and the
tuned run always score the same records — differing counts would make the
comparison meaningless with nothing to warn you.
"""),
    code("""
MODEL   = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"  # ungated 4-bit mirror
ADAPTER = "training/adapters/omniserve-slm-8b"

LIMIT   = 200    # test records to score; None for the full 500 (+~30 min)
EPOCHS  = 1      # 1 fits a free T4; use 2 on L4/A100
MAX_SEQ = 1024   # longest example is ~640 tokens
BATCH   = 8      # eval batch size

_limit = f"--limit {LIMIT}" if LIMIT else ""
print(f"{MODEL}\\n{EPOCHS} epoch(s), seq {MAX_SEQ}, scoring {LIMIT or 500} records")
"""),
    md("## Dataset"),
    code("""
!python -m data.generate_dataset --count 10000 --offline --noise 0.01
!python -m data.split_dataset
"""),
    md("""
## Baseline

Measured before training, so nothing downstream can influence it.

The baseline gets `--include-schema`; the tuned model won't. It needs the schema
to know what fields to emit, and that asymmetry favours the baseline — so the
improvement comes out understated rather than inflated.
"""),
    code("""
!python -m benchmarks.eval_local --model {MODEL} --tag baseline \\
    --include-schema --batch-size {BATCH} {_limit}
"""),
    md("""
### What did the baseline actually say?

A near-zero F1 can mean the model is bad — or that it answered in a shape the
scorer doesn't recognise. Read a few raw outputs before believing the number.
Untuned models handed a schema often echo the schema back, which parses as JSON
and validates as nothing.
"""),
    code("""
import json

for line in list(open("benchmarks/results/predictions-baseline.jsonl"))[:3]:
    row = json.loads(line)
    print(f"--- {row['id']} ---")
    print(row["prediction"][:400])
    print()
"""),
    md("""
## Train

Watch the loss fall from ~1.5 toward <0.3. Flat loss means something is wrong —
stop and check the data cell output.
"""),
    code("""
!python -m training.train_qlora --model {MODEL} \\
    --max-seq-length {MAX_SEQ} --epochs {EPOCHS} --output {ADAPTER}
"""),
    md("""
## Tuned

No `--include-schema`: the model learned the schema, and omitting it saves ~400
tokens of prefill per request at serve time.
"""),
    code("""
!python -m benchmarks.eval_local --model {MODEL} --adapter {ADAPTER} \\
    --tag tuned --batch-size {BATCH} {_limit}
"""),
    md("## Results"),
    code("""
import json
from pathlib import Path

b = json.loads(Path("benchmarks/results/accuracy-baseline.json").read_text())
t = json.loads(Path("benchmarks/results/accuracy-tuned.json").read_text())

print(f"{'':<26}{'baseline':>10}{'tuned':>10}{'delta':>10}")
print("-" * 56)
for label, key in [
    ("Field F1", "field_f1"),
    ("Field precision", "field_precision"),
    ("Field recall", "field_recall"),
    ("Exact match rate", "exact_match_rate"),
    ("Schema validity rate", "schema_validity_rate"),
    ("Invalid JSON rate", "invalid_json_syntax_rate"),
]:
    print(f"{label:<26}{b[key]:>10.4f}{t[key]:>10.4f}{t[key] - b[key]:>+10.4f}")
"""),
    md("""
This table isolates what *fine-tuning* contributed. Constrained decoding is not
active here — it lands at serve time and takes schema validity to 1.0 by itself,
so crediting it to the fine-tune would overstate both.
"""),
    md("## Save"),
    code("""
from google.colab import drive

drive.mount("/content/drive")
!mkdir -p "/content/drive/MyDrive/omniserve"
!cp -r {ADAPTER} "/content/drive/MyDrive/omniserve/"
!cp benchmarks/results/*.json "/content/drive/MyDrive/omniserve/"
!ls -la "/content/drive/MyDrive/omniserve/"
"""),
    md("""
Copy `accuracy-baseline.json` and `accuracy-tuned.json` back into the repo —
Phase 7 charts against them.
"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

path = Path("training/colab_finetune.ipynb")
path.write_text(json.dumps(notebook, indent=1) + "\n")
print(f"wrote {path} ({len(cells)} cells)")
