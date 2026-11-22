"""Download TinyShakespeare, build a character vocabulary, write token files (AI-scaffolded).

Run from anywhere:  python data/prepare.py
Outputs (next to this file): input.txt, train.bin, val.bin, meta.json
"""
import json
import urllib.request
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
TRAIN_FRAC = 0.9  # first 90% of the text trains, last 10% validates (no shuffling)


def main() -> None:
    # 1) Get the raw text (skip the download if we already have it).
    txt_path = DATA_DIR / "input.txt"
    if not txt_path.exists():
        print(f"Downloading {URL} ...")
        urllib.request.urlretrieve(URL, txt_path)
    text = txt_path.read_text(encoding="utf-8")
    print(f"Characters in corpus: {len(text):,}")

    # 2) Character-level "tokenizer": every distinct character gets an integer id.
    #    Sorting makes the mapping deterministic across machines.
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    print(f"Vocab size: {vocab_size}")
    if vocab_size != 65:
        print("WARNING: expected 65 characters for TinyShakespeare; "
              "check the download (and Config.vocab_size).")

    # 3) Encode the whole corpus as uint16 (vocab is far below 65,536).
    ids = np.array([stoi[c] for c in text], dtype=np.uint16)

    # 4) Contiguous train/val split. Validation text is never seen during training.
    n_train = int(TRAIN_FRAC * len(ids))
    ids[:n_train].tofile(DATA_DIR / "train.bin")
    ids[n_train:].tofile(DATA_DIR / "val.bin")

    # 5) Save the vocabulary so ids can be decoded back to text later.
    meta = {
        "chars": chars,
        "vocab_size": vocab_size,
        "n_train": int(n_train),
        "n_val": int(len(ids) - n_train),
    }
    (DATA_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"train tokens: {meta['n_train']:,} | val tokens: {meta['n_val']:,}")
    print(f"Wrote train.bin, val.bin, meta.json to {DATA_DIR}")


if __name__ == "__main__":
    main()
