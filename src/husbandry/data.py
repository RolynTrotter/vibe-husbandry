"""Tokenizers, corpus loading, and batch samplers.

Corpora are tokenized once into a flat token file (``data/<name>/{train,val}.bin``)
and memory-mapped, nanoGPT style. Sources:

- ``hf:<dataset_id>``   a Hugging Face dataset (``text_field`` column, ``split``)
- ``file:<glob>``       local text files, one document per file
- ``synth:<task>``      a synthetic skill from ``husbandry.tasks`` (``max_docs`` documents)

A run trains on a weighted mixture of corpora. All of them use one tokenizer, which is
saved into the run directory: descendants of a run always reuse it (species rule).
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

EOS = "<|endoftext|>"


@dataclass
class DataConfig:
    source: str = "hf:roneneldan/TinyStories"
    weight: float = 1.0  # sampling weight within the run's mixture
    split: str = "train"
    text_field: str = "text"
    max_docs: int | None = 200_000  # cap so a toy corpus tokenizes in minutes
    val_fraction: float = 0.01
    cache_dir: str = "data"

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def name(self) -> str:
        return self.source.split(":", 1)[-1].split("/")[-1].replace("*", "x")[:40]


# ---------------------------------------------------------------- tokenizers


class ByteTokenizer:
    """UTF-8 bytes plus one EOS token. No training, no downloads: good for tests."""

    vocab_size = 257
    eos_id = 256
    fingerprint = "byte"

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [self.encode(t) for t in texts]

    def decode(self, ids: list[int]) -> str:
        return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")

    def save(self, run_dir: Path) -> None:
        (run_dir / "tokenizer_meta.json").write_text(json.dumps({"kind": "byte"}))


class HFTokenizer:
    """Wraps a ``tokenizers.Tokenizer`` (trained locally or pulled from the Hub)."""

    def __init__(self, tok, eos_token: str = EOS):
        self.tok = tok
        self.eos_token = eos_token
        self.eos_id = tok.token_to_id(eos_token)
        if self.eos_id is None:
            raise ValueError(f"tokenizer has no {eos_token!r} token")
        self.vocab_size = tok.get_vocab_size()
        self.fingerprint = hashlib.sha1(tok.to_str().encode()).hexdigest()[:12]

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [e.ids for e in self.tok.encode_batch(texts)]

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids)

    def save(self, run_dir: Path) -> None:
        self.tok.save(str(run_dir / "tokenizer.json"))
        meta = {"kind": "hf", "eos": self.eos_token}
        (run_dir / "tokenizer_meta.json").write_text(json.dumps(meta))


def tokenizer_from_run(run_dir: str | Path):
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "tokenizer_meta.json").read_text())
    if meta["kind"] == "byte":
        return ByteTokenizer()
    from tokenizers import Tokenizer

    return HFTokenizer(Tokenizer.from_file(str(run_dir / "tokenizer.json")), meta["eos"])


def build_tokenizer(spec: str, corpus: DataConfig):
    """``byte`` | ``bpe:<vocab>`` (trained on ``corpus``) | ``hf:<repo>`` | ``run:<dir>``."""
    kind, _, ref = spec.partition(":")
    if kind == "byte":
        return ByteTokenizer()
    if kind == "run":
        return tokenizer_from_run(ref)
    from tokenizers import Tokenizer

    if kind == "hf":
        tok = Tokenizer.from_pretrained(ref)
        return HFTokenizer(tok, EOS if tok.token_to_id(EOS) is not None else "</s>")
    if kind == "bpe":
        key = json.dumps({"spec": spec, "corpus": corpus.to_dict()}, sort_keys=True)
        h = hashlib.sha1(key.encode()).hexdigest()[:10]
        path = Path(corpus.cache_dir) / "tokenizers" / f"bpe{ref}-{corpus.name}-{h}.json"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            _train_bpe(corpus, int(ref), path)
        return HFTokenizer(Tokenizer.from_file(str(path)))
    raise ValueError(f"unknown tokenizer {spec!r}")


def _train_bpe(cfg: DataConfig, vocab_size: int, path: Path) -> None:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOS],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(_iter_texts(cfg), trainer=trainer)
    tok.save(str(path))


# ---------------------------------------------------------------- corpora


def _iter_texts(cfg: DataConfig) -> Iterator[str]:
    kind, _, ref = cfg.source.partition(":")
    limit = cfg.max_docs
    if kind == "file":
        paths = sorted(glob.glob(ref, recursive=True))
        if not paths:
            raise FileNotFoundError(f"no files match {ref!r}")
        for i, p in enumerate(paths):
            if limit is not None and i >= limit:
                return
            yield Path(p).read_text(encoding="utf-8", errors="replace")
    elif kind == "hf":
        from datasets import load_dataset

        ds = load_dataset(ref, split=cfg.split, streaming=True)
        for i, row in enumerate(ds):
            if limit is not None and i >= limit:
                return
            yield row[cfg.text_field]
    elif kind == "synth":
        from husbandry.tasks import train_docs

        if limit is None:
            raise ValueError("synthetic sources need max_docs")
        yield from train_docs(ref, limit)
    else:
        raise ValueError(f"unknown data source {cfg.source!r}")


def corpus_dir(cfg: DataConfig, tok) -> Path:
    """Cache directory keyed by everything that changes the token stream."""
    fields = ("source", "split", "text_field", "max_docs", "val_fraction")
    key_d = {k: getattr(cfg, k) for k in fields} | {"tok": tok.fingerprint}
    if cfg.source.startswith("synth:"):  # a task's format can change; key on its output
        from husbandry.tasks import train_docs

        key_d["sample"] = next(train_docs(cfg.source.split(":", 1)[1], 1))
    key = json.dumps(key_d)
    return Path(cfg.cache_dir) / f"{cfg.name}-{hashlib.sha1(key.encode()).hexdigest()[:10]}"


def prepare(cfg: DataConfig, tok) -> Path:
    """Tokenize the corpus once with ``tok``; returns the directory with train/val .bin."""
    out = corpus_dir(cfg, tok)
    meta_path = out / "meta.json"
    if meta_path.exists():
        return out
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
        "tokenizer": tok.fingerprint,
        "dtype": np.dtype(dtype).name,
        "vocab_size": tok.vocab_size,
        "n_docs": n_docs,
        "n_tokens": {"train": len(arr) - n_val, "val": n_val},
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return out


def load_tokens(directory: Path, split: str) -> np.ndarray:
    meta = json.loads((directory / "meta.json").read_text())
    return np.memmap(directory / f"{split}.bin", dtype=meta["dtype"], mode="r")


# ---------------------------------------------------------------- samplers


class BatchSampler:
    """Uniformly random windows of ``seq_len + 1`` tokens."""

    def __init__(self, tokens: np.ndarray, batch_size: int, seq_len: int, seed: int = 0):
        if len(tokens) <= seq_len + 1:
            raise ValueError(f"corpus has {len(tokens)} tokens; need > seq_len + 1 = {seq_len + 1}")
        self.tokens, self.batch_size, self.seq_len = tokens, batch_size, seq_len
        self.rng = np.random.default_rng(seed)

    def windows(self, n: int) -> np.ndarray:
        starts = self.rng.integers(0, len(self.tokens) - self.seq_len - 1, n)
        return np.stack([self.tokens[s : s + self.seq_len + 1] for s in starts]).astype(np.int64)

    def sample(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        return _to_device(self.windows(self.batch_size), device)


class MixtureSampler:
    """Each row of a batch comes from corpus ``i`` with probability ``weights[i]``."""

    def __init__(self, samplers: list[BatchSampler], weights: list[float], seed: int = 0):
        w = np.asarray(weights, dtype=float)
        if len(samplers) != len(w) or (w <= 0).any():
            raise ValueError("need one positive weight per corpus")
        self.samplers, self.p = samplers, w / w.sum()
        self.batch_size = samplers[0].batch_size
        self.rng = np.random.default_rng(seed)

    def sample(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        counts = self.rng.multinomial(self.batch_size, self.p)
        rows = [s.windows(c) for s, c in zip(self.samplers, counts, strict=True) if c]
        win = np.concatenate(rows)
        return _to_device(win[self.rng.permutation(len(win))], device)


def _to_device(win: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    t = torch.from_numpy(win)
    x, y = t[:, :-1], t[:, 1:]
    if device.type == "cuda":
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )
    return x.to(device), y.to(device)
