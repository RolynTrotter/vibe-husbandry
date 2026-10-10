#!/usr/bin/env bash
# One full toy breeding cycle plus compute-matched baselines, then a comparison table.
#
#   founder ──┬── dam  (specialise: arithmetic) ──┐
#             └── sire (specialise: reverse)  ────┴── conceive ── embryo (8 experts)
#   embryo ── gestate (router warm-up, then experts) ── child ── prune to 4 ── adults
#
# Baselines: longer (founder trained on the mixture for the same extra tokens),
# direct (MoE-4 from scratch on the mixture, same total tokens), avg / ties (weight-space
# merges of dam and sire).
#
# Usage: scripts/toy_cycle.sh                      # full size (~45 min on a Mac Studio)
#        F=300 P=100 W=20 G=100 scripts/toy_cycle.sh   # quick shake-down
# Env: RUNS (output dir, default runs), F/P/W/G (founder / parent / router warm-up /
#      gestation steps), LAMBDA (trunk task-arithmetic weight at conception), EXTRA ("--set k=v ..." applied to every train call),
#      FORCE=1 to redo stages whose run directory already exists.
set -euo pipefail
cd "$(dirname "$0")/.."
RUNS=${RUNS:-runs}
F=${F:-2000} P=${P:-600} W=${W:-100} G=${G:-500}
EXTRA=${EXTRA:-}
EXTRA_EVAL=${EXTRA_EVAL:-}  # e.g. "--set data.cache_dir=\"x\""
H="poetry run husbandry"
mkdir -p "$RUNS"

done_() { [ -z "${FORCE:-}" ] && [ -f "$RUNS/$1/lineage.json" ] && echo "== $1: exists, skipping"; }
train() { # name config steps [init_from]
  done_ "$1" && return
  echo "== train $1"
  local init=()
  [ -n "${4:-}" ] && init=(--set "train.init_from=\"$RUNS/$4\"")
  # shellcheck disable=SC2086
  $H train "configs/toy/$2.toml" --set "train.out_dir=\"$RUNS\"" --set "train.max_steps=$3" \
    "${init[@]}" $EXTRA
}

train founder founder "$F"
train dam dam "$P" founder
train sire sire "$P" founder

# trunk = founder + LAMBDA * (tau_dam + tau_sire); 0.5 is a plain average
done_ embryo || $H conceive "$RUNS/dam" "$RUNS/sire" --name embryo --out "$RUNS" \
  --base "$RUNS/founder" --trunk-lambda "${LAMBDA:-0.5}"
train child-warm child-warm "$W" embryo
train child child "$G" child-warm
for m in reap allele random; do
  done_ "adult-$m" || $H prune "$RUNS/child" --name "adult-$m" --keep 4 --method "$m" --out "$RUNS"
done

train longer longer $((2 * P + W + G)) founder
# direct uses the founder's tokenizer (tokenizer = "run:runs/founder" in its config)
# shellcheck disable=SC2086
done_ direct || $H train configs/toy/direct.toml --set "train.out_dir=\"$RUNS\"" \
  --set "tokenizer=\"run:$RUNS/founder\"" --set "train.max_steps=$((F + 2 * P + W + G))" $EXTRA
done_ avg || $H merge "$RUNS/dam" "$RUNS/sire" --name avg --method average --out "$RUNS"
done_ ties || $H merge "$RUNS/dam" "$RUNS/sire" --name ties --method ties \
  --base "$RUNS/founder" --out "$RUNS"

ALL=(founder dam sire embryo child adult-reap adult-allele adult-random longer direct avg ties)
DIRS=("${ALL[@]/#/$RUNS/}")
# shellcheck disable=SC2086
$H evaluate "${DIRS[@]}" --lm-config configs/toy/founder.toml $EXTRA_EVAL
$H report "${DIRS[@]}" | tee "$RUNS/report.md"
