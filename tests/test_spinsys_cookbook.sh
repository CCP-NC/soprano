#!/usr/bin/env bash
# test_spinsys_cookbook.sh — run all spinsys cookbook examples and inspect output.
# Usage: bash tests/test_spinsys_cookbook.sh [magres_file]
# Defaults to tests/test_data/ethanol.magres
# In dev you want to run this with hatch e.g.:
# uvx hatch run hatch-test.py3.13:bash tests/test_spinsys_cookbook.sh

set -euo pipefail

MAGRES="${1:-tests/test_data/ethanol.magres}"
OUT="$(mktemp -d)"
PASS=0
FAIL=0

sep() { echo ""; echo "══════════════════════════════════════════════════"; echo "  $*"; echo "══════════════════════════════════════════════════"; }
ok()  { echo "  ✓ $*"; PASS=$((PASS+1)); }
err() { echo "  ✗ $*"; FAIL=$((FAIL+1)); }

check_simpson() {
    local label="$1" file="$2" expected_spinsys="$3"
    if [[ ! -f "$file" ]]; then
        err "$label: output file not created"
        return
    fi
    if grep -q "spinsys" "$file"; then
        ok "$label: contains 'spinsys' block"
    else
        err "$label: missing 'spinsys' block"
        cat "$file"
    fi
    if [[ -n "$expected_spinsys" ]]; then
        if grep -q "$expected_spinsys" "$file"; then
            ok "$label: contains expected content '$expected_spinsys'"
        else
            err "$label: missing expected content '$expected_spinsys'"
            cat "$file"
        fi
    fi
}

check_json() {
    local label="$1" file="$2"
    if [[ ! -f "$file" ]]; then
        err "$label: output file not created"
        return
    fi
    if python -c "import json,sys; json.load(open('$file'))" 2>/dev/null; then
        ok "$label: valid JSON"
    else
        err "$label: invalid JSON"
        cat "$file"
    fi
}

echo "Using magres file: $MAGRES"
echo "Output dir: $OUT"

# ── 1. Basic 1H spin system → Simpson ────────────────────────────────────────
sep "1. Basic 1H → Simpson"
soprano spinsys "$MAGRES" -s H --ref H:30 -o "$OUT/01_H.in"
check_simpson "1H Simpson" "$OUT/01_H.in" "shift"
echo "--- output ---"; cat "$OUT/01_H.in"

# ── 2. 13C spin system with 1H dipolar couplings ─────────────────────────────
sep "2. 13C/1H + C-H dipolar → Simpson"
soprano spinsys "$MAGRES" -s C,H --ref C:170,H:30 --dip --select-i C --select-j H -o "$OUT/02_C_dip.in"
check_simpson "13C+dip Simpson" "$OUT/02_C_dip.in" "dipole"
echo "--- output ---"; cat "$OUT/02_C_dip.in"

# ── 3. Quadrupolar 2H (2nd order) → Simpson ──────────────────────────────────
sep "3. 2H quadrupolar (q-order 2) → Simpson"
soprano spinsys "$MAGRES" -s H -i 2H --ref H:0 --q-order 2 -o "$OUT/03_2H_quad.in"
check_simpson "2H quad Simpson" "$OUT/03_2H_quad.in" "quadrupole"
echo "--- output ---"; cat "$OUT/03_2H_quad.in"

# ── 4. 17O with EFG + observed nucleus → Simpson ─────────────────────────────
sep "4. 17O EFG + --obs → Simpson"
soprano spinsys "$MAGRES" -s O -i 17O --q-order 2 --obs 17O --no-ms -o "$OUT/04_17O.in"
check_simpson "17O Simpson" "$OUT/04_17O.in" "quadrupole"
echo "--- output ---"; cat "$OUT/04_17O.in"

# ── 5. J-couplings ───────────────────────────────────────────────────────────
sep "5. 13C + dipolar + J-couplings → Simpson"
soprano spinsys "$MAGRES" -s C --ref C:170 --dip --jcoupling -o "$OUT/05_C_J.in"
check_simpson "C+J Simpson" "$OUT/05_C_J.in" "jcoupling"
echo "--- output ---"; cat "$OUT/05_C_J.in"

# ── 6. MRSimulator format ─────────────────────────────────────────────────────
sep "6. 13C + dipolar → MRSimulator JSON"
soprano spinsys "$MAGRES" -s C --ref C:170 --dip -f mrsimulator -o "$OUT/06_C_mrs.json"
check_json "MRSimulator JSON" "$OUT/06_C_mrs.json"
echo "--- output ---"; cat "$OUT/06_C_mrs.json"

# ── 7. --split ───────────────────────────────────────────────────────────────
sep "7. --split (one file per H site)"
soprano spinsys "$MAGRES" -s H --ref H:30 --split -o "$OUT/07_split.in"
N_SPLIT=$(ls "$OUT"/07_split_*.in 2>/dev/null | wc -l | tr -d ' ')
if [[ "$N_SPLIT" -gt 1 ]]; then
    ok "--split: created $N_SPLIT files"
else
    err "--split: expected multiple files, got $N_SPLIT"
fi
ls "$OUT"/07_split_*.in 2>/dev/null

# ── 8. CH3 averaging + --no-reduce ───────────────────────────────────────────
sep "8. CH3 averaging + --no-reduce"
soprano spinsys "$MAGRES" -s H --ref H:30 --average-group CH3 --no-reduce -o "$OUT/08_CH3.in"
check_simpson "CH3 avg Simpson" "$OUT/08_CH3.in" "shift"
echo "--- output ---"; cat "$OUT/08_CH3.in"

# ── 9. Custom isotope: 2H dipolar (no MS) ─────────────────────────────────────
sep "9. Custom isotope 2H, dipolar only (no MS)"
soprano spinsys "$MAGRES" -s H -i 2H --dip --no-ms -o "$OUT/09_2H_dip.in"
check_simpson "2H dip only" "$OUT/09_2H_dip.in" "dipole"
# Should NOT contain 'shift' (no --ref given)
if grep -q "shift" "$OUT/09_2H_dip.in"; then
    err "9. unexpected 'shift' in dipolar-only output"
else
    ok "9. no 'shift' in dipolar-only output (correct — no --ref given)"
fi
echo "--- output ---"; cat "$OUT/09_2H_dip.in"

# ── 10. Symmetry reduction: --mean-merge ──────────────────────────────────────
sep "10. --mean-merge symmetry reduction"
soprano spinsys "$MAGRES" -s H --ref H:30 --mean-merge -o "$OUT/10_mean_merge.in"
check_simpson "mean-merge Simpson" "$OUT/10_mean_merge.in" "shift"
echo "--- output ---"; cat "$OUT/10_mean_merge.in"

# ── Summary ───────────────────────────────────────────────────────────────────
sep "Summary"
echo "  PASSED: $PASS"
echo "  FAILED: $FAIL"
echo "  Output files in: $OUT"
[[ $FAIL -eq 0 ]] && exit 0 || exit 1
