#!/usr/bin/env bash
# Runs every shortcut baseline through the sealed verifier and records the
# outcome.  Authoring evidence only; never executed by the harness.
set -uo pipefail
MODES="reference all_zero transport_from_residual deposition_omitted \
deposition_double_counted moist_air_mass mask_term_omitted mask_term_start_state \
end_of_step_weights radical_diagnostic_as_production drop_o1d_channel \
duplicate_titration scenarios_swapped threshold_after_integration \
receptor_nearest_cell peak_of_difference interaction_always_positive \
dominant_ignores_mask enhancement_means_production sustain_from_enhancement_only"

printf '%-38s %s\n' "baseline" "reward"
for m in nop $MODES; do
  rm -rf /app/output; mkdir -p /app/output
  if [ "$m" != "nop" ]; then
    BASELINE_MODE="$m" python3 "$(dirname "$0")/baselines.py" >/dev/null 2>&1 || true
  fi
  bash "$(dirname "$0")/../../tests/test.sh" >/dev/null 2>&1
  printf '%-38s %s\n' "$m" "$(cat /logs/verifier/reward.txt)"
done
