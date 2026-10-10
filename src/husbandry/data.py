"""Tokenizers, corpus loading, and a random-window batch sampler.

Corpora are tokenized once into a flat token file (``data/<name>/{train,val}.bin``)
and memory-mapped, nanoGPT style. Sources:

- ``hf:<dataset_id>``   a Hugging Face dataset (``text_field`` column, ``split``)
- ``file:<glob>``       local text files, one document per file
"""

from __future__ import annotations

import glob
import hashlib
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class DataConfig:
    source: str = "hf:roneneldan/TinyStories"
    split: str = "train"
    text_field: str = "text"
    max_docs: int | None = 200_000  # cap so a toy corpus tokenizes in minutes
    tokenizer: str = "bpe:4096"  # byte | bpe:<vocab> | hf:<repo>
    val_fraction: float = 0.01
    cache_dir: str = "data"

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- tokenizers


class ByteTokenizer:
    """UTF-8 bytes plus one EOS token. No training, no downloads: good for tests."""

    vocab_size = 257
    eos_id = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [self.encode(t) for t in texts]

    def decode(self, ids: list[int]) -> str:
        return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")


class HFTokenizer:
    """Wraps a ``tokenizers.Tokenizer`` (trained locally or pulled from the Hub)."""

    def __init__(self, tok, eos_token: str):
        self.tok = tok
        self.eos_id = tok.token_to_id(eos_token)
        if self.eos_id is None:
            raise ValueError(f"tokenizer has no {eos_token!r} token")
        self.vocab_size = tok.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [e.ids for e in self.tok.encode_batch(texts)]

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids)


def _iter_texts(cfg: DataConfig) -> Iterator[str]:
    kind, _, ref = cfg.source.partition(":")
    if kind == "file":
        paths = sorted(glob.glob(ref, recursive=True))
        if not paths:
            raise FileNotFoundError(f"no files match {ref!r}")
        for i, p in enumerate(paths):
            if cfg.max_docs is not None and i >= cfg.max_docs:
                return
            yield Path(p).read_text(encoding="utf-8", errors="replace")
    elif kind == "hf":
        from datasets import load_dataset

        ds = load_dataset(ref, split=cfg.split, streaming=True)
        for i, row in enumerate(ds):
            if cfg.max_docs is not None and i >= cfg.max_docs:
                return
            yield row[cfg.text_field]
    else:
        raise ValueError(f"unknown data source {cfg.source!r}")


def corpus_dir(cfg: DataConfig) -> Path:
    """Cache directory keyed by everything that changes the token stream."""
    key = json.dumps(
        {k: getattr(cfg, k) for k in ("source", "split", "text_field", "max_docs", "tokenizer")},
        sort_keys=True,
    )
    slug = cfg.source.split(":", 1)[-1].replace("/", "_").replace("*", "x")[:40]
    return Path(cfg.cache_dir) / f"{slug}-{hashlib.sha1(key.encode()).hexdigest()[:10]}"


def load_tokenizer(cfg: DataConfig):
    kind, _, ref = cfg.tokenizer.partition(":")
    if kind == "byte":
        return ByteTokenizer()
    from tokenizers import Tokenizer

    if kind == "hf":
        tok = Tokenizer.from_pretrained(ref)
        eos = "<|endoftext|>" if tok.token_to_id("<|endoftext|>") is not None else "</s>"
        return HFTokenizer(tok, eos)
    if kind == "bpe":
        path = corpus_dir(cfg) / "tokenizer.json"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            _train_bpe(cfg, int(ref), path)
        return HFTokenizer(Tokenizer.from_file(str(path)), "<|endoftext|>")
    raise ValueError(f"unknown tokenizer {cfg.tokenizer!r}")


def _train_bpe(cfg: DataConfig, vocab_size: int, path: Path) -> None:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(_iter_texts(cfg), trainer=trainer)
    tok.save(str(path))


# ---------------------------------------------------------------- token files


def prepare(cfg: DataConfig) -> tuple[Path, object]:
    """Tokenize the corpus once; returns (directory with train/val .bin, tokenizer)."""
    tok = load_tokenizer(cfg)
    out = corpus_dir(cfg)
    meta_path = out / "meta.json"
    if meta_path.exists():
        return out, tok
    out.mkdir(parents=True, exist_ok=True)
    dtype = np.uint16 if tok.vocab_size <= 2**16 else np.uint32
    chunks: list[np.ndarray] = []
    batch: list[str] = []
    n_docs = 0

    def flush() -> None:
        for ids in tok.encode_batch(batch):
            chunks.append(np.asarray([*ids, tok.eos_id], dtype=dtype))
        batch.clear()

    for text in _iter_texts(cfg):
        batch.append(text)
        n_docs += 1
        if len(batch) >= 1024:
            flush()
    flush()
    arr = np.concatenate(chunks)
    n_val = max(int(len(arr) * cfg.val_fraction), 1)
    arr[:-n_val].tofile(out / "train.bin")
    arr[-n_val:].tofile(out / "val.bin")
    meta = {
        "config": cfg.to_dict(),
        "dtype": np.dtype(dtype).name,
        "vocab_size": tok.vocab_size,
        "n_docs": n_docs,
        "n_tokens": {"train": len(arr) - n_val, "val": n_val},
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return out, tok


def load_tokens(directory: Path, split: str) -> np.ndarray:
    meta = json.loads((directory / "meta.json").read_text())
    return np.memmap(directory / f"{split}.bin", dtype=meta["dtype"], mode="r")


class BatchSampler:
    """Uniformly random windows of ``seq_len + 1`` tokens."""

    def __init__(self, tokens: np.ndarray, batch_size: int, seq_len: int, seed: int = 0):
        if len(tokens) <= seq_len + 1:
            raise ValueError(f"corpus has {len(tokens)} tokens; need > seq_len + 1 = {seq_len + 1}")
        self.tokens, self.batch_size, self.seq_len = tokens, batch_size, seq_len
        self.rng = np.random.default_rng(seed)

    def sample(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        starts = self.rng.integers(0, len(self.tokens) - self.seq_len - 1, self.batch_size)
        win = np.stack([self.tokens[s : s + self.seq_len + 1] for s in starts]).astype(np.int64)
        win = torch.from_numpy(win)
        x, y = win[:, :-1], win[:, 1:]
        if device.type == "cuda":
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
                device, non_blocking=True
            )
        return x.to(device), y.to(device)
