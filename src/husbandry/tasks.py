"""Synthetic skills with exact-match answers, for the toy breeding cycle.

Each task makes (prompt, answer) items. Training documents are newline-joined
``prompt + answer`` lines; evaluation greedily completes prompts. For skills that
generalise, an item is held out iff ``crc32(prompt) % 10 == 0``, so training generators
can skip eval items without storing any list. ``facts`` is pure memory: there is nothing
to generalise to, so it is evaluated on the facts it was taught.

- ``facts``    "fact: blorvat -> cake"  (a fixed table of 1000 made-up names)

- ``arith``    "47+85=231", "92-17=570"  (2-digit operands, 3-digit answer written
               least-significant digit first; ``arith3``: 3-digit operands)
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
    """Fixed-width operands and a fixed-width answer written least-significant digit first
    ("02+93=590" means 095). Small transformers learn carries far faster this way, because
    every digit sits at a fixed offset (Lee et al. 2023, arXiv 2307.03381)."""
    w = len(str(limit - 1))

    def make(rng: random.Random) -> tuple[str, str]:
        a, b = rng.randrange(limit), rng.randrange(limit)
        if rng.random() < 0.5:
            op, c = "+", a + b
        else:
            a, b = max(a, b), min(a, b)
            op, c = "-", a - b
        return f"{a:0{w}d}{op}{b:0{w}d}=", f"{c:0{w + 1}d}"[::-1]

    return make


def _reverse(rng: random.Random) -> tuple[str, str]:
    words = [rng.choice(WORDS) for _ in range(rng.randint(3, 6))]
    return f"reverse: {' '.join(words)} ->", " " + " ".join(reversed(words))


_SYL = "ba be bi bo bu da de di do du ka ke ki ko ku la le li lo lu ma me mi mo mu na ne ni no nu ra re ri ro ru sa se si so su ta te ti to tu va ve vi vo vu za ze zi zo zu".split()  # noqa: E501


def _fact_table(n: int = 1000) -> dict[str, str]:
    rng = random.Random("facts-table")
    table: dict[str, str] = {}
    while len(table) < n:
        name = "".join(rng.choice(_SYL) for _ in range(rng.randint(2, 3)))
        table.setdefault(name, rng.choice(WORDS))
    return table


FACTS = _fact_table()
_FACT_NAMES = sorted(FACTS)


def _facts(rng: random.Random) -> tuple[str, str]:
    name = rng.choice(_FACT_NAMES)
    return f"fact: {name} ->", " " + FACTS[name]


MEMORY_TASKS = {"facts"}  # evaluated on taught items: no held-out split

TASKS: dict[str, Callable[[random.Random], tuple[str, str]]] = {
    "facts": _facts,
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
            if task in MEMORY_TASKS or not is_heldout(prompt):
                lines.append(prompt + answer)
        yield "\n".join(lines) + "\n"


def eval_items(task: str, n: int, seed: int = 0) -> list[tuple[str, str]]:
    """``n`` distinct held-out items (taught items for memory tasks); the same list for
    the same (task, n, seed)."""
    make = TASKS[task]
    rng = random.Random(f"{task}-eval-{seed}")
    items: dict[str, str] = {}
    while len(items) < n:
        prompt, answer = make(rng)
        if task in MEMORY_TASKS or is_heldout(prompt):
            items.setdefault(prompt, answer)
    return list(items.items())
