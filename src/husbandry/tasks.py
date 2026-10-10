"""Synthetic skills with exact-match answers, for the toy breeding cycle.

Each task makes (prompt, answer) items. Training documents are newline-joined
``prompt + answer`` lines; evaluation greedily completes held-out prompts.
An item is held out iff ``crc32(prompt) % 10 == 0``, so training generators can
skip eval items without storing any list.

- ``arith``    "47+85=231", "92-17=57"  (operands < 100, answer digits reversed;
               ``arith3``: operands < 1000)
- ``reverse``  "reverse: cat dog sun -> sun dog cat"
"""

from __future__ import annotations

import random
import zlib
from collections.abc import Callable, Iterator

WORDS = (
    "cat dog sun moon tree ball bird fish cake milk boat star hat car bed cup "
    "red blue big small happy sad fast slow jump run play sing eat sleep walk swim "
    "girl boy mom dad friend house park toy book door rain snow wind fire box key"
).split()


def _arith_upto(limit: int) -> Callable[[random.Random], tuple[str, str]]:
    """Answers are written least-significant digit first ("47+85=231" means 132): small
    transformers learn carries far faster this way (Lee et al. 2023, arXiv 2307.03381)."""

    def make(rng: random.Random) -> tuple[str, str]:
        a, b = rng.randrange(limit), rng.randrange(limit)
        if rng.random() < 0.5:
            return f"{a}+{b}=", str(a + b)[::-1]
        a, b = max(a, b), min(a, b)
        return f"{a}-{b}=", str(a - b)[::-1]

    return make


def _reverse(rng: random.Random) -> tuple[str, str]:
    words = [rng.choice(WORDS) for _ in range(rng.randint(3, 6))]
    return f"reverse: {' '.join(words)} ->", " " + " ".join(reversed(words))


TASKS: dict[str, Callable[[random.Random], tuple[str, str]]] = {
    "arith": _arith_upto(100),
    "arith3": _arith_upto(1000),
    "reverse": _reverse,
}


def is_heldout(prompt: str) -> bool:
    return zlib.crc32(prompt.encode()) % 10 == 0


def train_docs(task: str, n_docs: int, seed: int = 0, items_per_doc: int = 16) -> Iterator[str]:
    make = TASKS[task]
    rng = random.Random(f"{task}-train-{seed}")
    for _ in range(n_docs):
        lines = []
        while len(lines) < items_per_doc:
            prompt, answer = make(rng)
            if not is_heldout(prompt):
                lines.append(prompt + answer)
        yield "\n".join(lines) + "\n"


def eval_items(task: str, n: int, seed: int = 0) -> list[tuple[str, str]]:
    """``n`` distinct held-out items; the same list for the same (task, n, seed)."""
    make = TASKS[task]
    rng = random.Random(f"{task}-eval-{seed}")
    items: dict[str, str] = {}
    while len(items) < n:
        prompt, answer = make(rng)
        if is_heldout(prompt):
            items.setdefault(prompt, answer)
    return list(items.items())
