#!/usr/bin/env bash
# Run every experiment (tasks 1-6) and draw the figures.
#
# Run from the repo root on the HOST -- it dispatches each task into the right
# container itself:
#     bash scripts/experiments/run_all.sh            # everything
#     bash scripts/experiments/run_all.sh 1 2 5      # only those tasks
#
# Task 4 (VTM inter) is the slow one: VTM is a reference encoder. It defaults to
# 8 frames, which gives a real curve in ~30 min for both QP sets. For the full
# 64 frames of the original brief, set VTM_FRAMES=64 -- and budget hours.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PATH="/Applications/Docker.app/Contents/Resources/bin:${PATH}"
COMPOSE="${COMPOSE:-docker compose}"
run() { ${COMPOSE} run --rm --no-deps "$1" bash -lc "$2"; }

WANT=("$@")
want() {
  [[ ${#WANT[@]} -eq 0 ]] && return 0
  local t; for t in "${WANT[@]}"; do [[ "$t" == "$1" ]] && return 0; done; return 1
}

mkdir -p outputs/experiments logs

if want 1; then
  echo "### Task 1 -- DCVC-UF image RD (kodim19, qp 0/15/30/45/63)"
  run image-compression "cd /opt/DCVC && python /work/scripts/experiments/exp1_dcvc_image_rd.py" \
    2>&1 | tee logs/exp1.log
fi

if want 2; then
  echo "### Task 2 -- VTM intra RD (kodim19, QP 22/27/32/37/42)"
  run hybrid-vtm "bash /work/scripts/experiments/exp2_vtm_intra_rd.sh" \
    2>&1 | tee logs/exp2.log
fi

if want 3; then
  echo "### Task 3 -- DCVC-UF video RD (RaceHorses 64 frames, qp 0/15/30/45/63)"
  run video-compression "cd /opt/DCVC && python /work/scripts/experiments/exp3_dcvc_video_rd.py --frames 64" \
    2>&1 | tee logs/exp3.log
fi

if want 4; then
  echo "### Task 4 -- VTM inter RD (RaceHorses, ${VTM_FRAMES:-8} frames)"
  # Both QP sets: the requested 0/15/30/45/63, and the standard 22..42 that
  # actually overlaps the DCVC rate range. See README_EXPERIMENTS.md.
  run hybrid-vtm "bash /work/scripts/experiments/exp4_vtm_inter_rd.sh --tag std --qps '22 27 32 37 42' --frames ${VTM_FRAMES:-8}" \
    2>&1 | tee logs/exp4_std.log
  run hybrid-vtm "bash /work/scripts/experiments/exp4_vtm_inter_rd.sh --tag requested --qps '0 15 30 45 63' --frames ${VTM_FRAMES:-8}" \
    2>&1 | tee logs/exp4_requested.log
fi

if want 5; then
  echo "### Task 5 -- AdaIN across styles + alpha sweep"
  run style-transfer "bash /work/scripts/experiments/exp5_adain_styles.sh" \
    2>&1 | tee logs/exp5.log
fi

if want 6; then
  echo "### Task 6 -- Real-ESRGAN vs classical resampling"
  run super-resolution "python /work/scripts/experiments/exp6_sr_compare.py" \
    2>&1 | tee logs/exp6.log

  echo "### Task 6b -- why the fidelity metrics and the eye disagree"
  # Depends on task 6's outputs, so it always follows it.
  run super-resolution "python /work/scripts/experiments/exp6b_perception_distortion.py" \
    2>&1 | tee logs/exp6b.log
fi

echo
echo "### Figures"
run image-compression "python /work/scripts/experiments/plot_rd.py all"
run style-transfer "python /work/scripts/experiments/make_figures.py all"

echo
echo "figures in outputs/experiments/figures/:"
ls -1 outputs/experiments/figures/ 2>/dev/null | sed 's|^|  |'
