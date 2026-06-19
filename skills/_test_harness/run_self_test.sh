#!/usr/bin/env bash
# Self-test for check_transcript.py.
# Verifies that the harness CORRECTLY passes a good transcript and CORRECTLY
# fails a transcript with lame bullets / truncated output / wrong flags.
#
# Usage: bash run_self_test.sh

set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
CHECKER="$HERE/check_transcript.py"
FIXTURES="$HERE/fixtures"

PYTHON="${PYTHON:-python3}"

pass_count=0
fail_count=0

run() {
  local name="$1"; shift
  local expected="$1"; shift   # "pass" or "fail"
  local fixture="$1"; shift

  echo "--- $name (expected: $expected) ---"
  set +e
  "$PYTHON" "$CHECKER" "$fixture" "$@"
  local rc=$?
  set -e

  if [[ "$expected" == "pass" && $rc -eq 0 ]] || [[ "$expected" == "fail" && $rc -eq 1 ]]; then
    echo "  >> SELF-TEST OK"
    pass_count=$((pass_count + 1))
  else
    echo "  >> SELF-TEST FAILED (rc=$rc)"
    fail_count=$((fail_count + 1))
  fi
  echo
}

# 1. Good transcript should PASS.
run "good_regress_transcript" pass "$FIXTURES/good_regress_transcript.txt" \
    --skill pyrsm-regress

# 2. Lame-bullets transcript should FAIL (and explain which checks failed).
run "bad_lame_bullets_transcript" fail "$FIXTURES/bad_lame_bullets_transcript.txt" \
    --skill pyrsm-regress

# 3. Truncated transcript should FAIL.
run "bad_truncated_transcript" fail "$FIXTURES/bad_truncated_transcript.txt" \
    --skill pyrsm-regress

# 4. PDP-misuse transcript should FAIL (relative importance claim from PDPs
#    without the required disclaimer + PIP redirect).
run "bad_pdp_misuse_transcript" fail "$FIXTURES/bad_pdp_misuse_transcript.txt" \
    --skill pyrsm-rforest

# 5. Good random-forest transcript with a proper importance disclaimer
#    should PASS — including the new IMPORTANCE_DISCLAIMER_PRESENT check.
run "good_rforest_with_disclaimer_transcript" pass \
    "$FIXTURES/good_rforest_with_disclaimer_transcript.txt" \
    --skill pyrsm-rforest

# 6. PIP fall-back-to-coefficients transcript should FAIL. The user asked
#    for PIP; the response interprets regression coefficients instead.
#    This is the exact Pi failure the user reported.
run "bad_pip_falls_back_to_coefs_transcript" fail \
    "$FIXTURES/bad_pip_falls_back_to_coefs_transcript.txt" \
    --prompt "$FIXTURES/prompt_pip_request.txt" \
    --skill pyrsm-regress

# 7. Good PIP transcript with verbatim DataFrame + PIP interpretation
#    should PASS.
run "good_pip_transcript" pass \
    "$FIXTURES/good_pip_transcript.txt" \
    --prompt "$FIXTURES/prompt_pip_request.txt" \
    --skill pyrsm-regress

# 8. PDP fall-back-to-coefficients transcript should FAIL. The user asked
#    for PDP plots; the response interprets regression coefficients without
#    saving plots or describing PDP shapes.
run "bad_pdp_falls_back_to_coefs_transcript" fail \
    "$FIXTURES/bad_pdp_falls_back_to_coefs_transcript.txt" \
    --prompt "$FIXTURES/prompt_pdp_request.txt" \
    --skill pyrsm-regress

# 9. Good PDP transcript with plot save + shape interpretation per panel
#    should PASS.
run "good_pdp_transcript" pass \
    "$FIXTURES/good_pdp_transcript.txt" \
    --prompt "$FIXTURES/prompt_pdp_request.txt" \
    --skill pyrsm-regress

echo "==============================="
echo "self-test results: $pass_count passed, $fail_count failed"
echo "==============================="

if [[ $fail_count -ne 0 ]]; then
  exit 1
fi
