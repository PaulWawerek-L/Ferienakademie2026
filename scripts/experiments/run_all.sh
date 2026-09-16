#!/usr/bin/env bash
# Run every experiment (tasks 1-8) and draw the figures.
#
# Run from the repo root on the HOST -- it dispatches each task into the right
# container itself:
#     bash scripts/experiments/run_all.sh            # everything
#     bash scripts/experiments/run_all.sh 1 2 5      # only those tasks
#
# Task 4 (VTM inter) is the slow one: VTM is a reference encoder. It defaults to
# 8 frames, which gives a real curve in ~30 min for both QP sets. For the full
# 64 frames of the original brief, set VTM_FRAMES=64 -- and budget hours.
#
# Exit status: 0 only if every step succeeded. Each step's real exit code is
# recorded -- `cmd | tee log` alone would report tee's -- and a summary of any
# failures is printed at the end. The script keeps going after a failure so one
# broken task does not cost the other forty minutes.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PATH="/Applications/Docker.app/Contents/Resources/bin:${PATH}"
COMPOSE="${COMPOSE:-docker compose}"
FAILED=()

# run <service> <command> <logfile>
run() {
  ${COMPOSE} run --rm --no-deps "$1" bash -lc "$2" 2>&1 | tee "$3"
  local rc=${PIPESTATUS[0]}
  [[ $rc -eq 0 ]] || FAILED+=("$3 (exit $rc)")
}

WANT=("$@")
want() {
  [[ ${#WANT[@]} -eq 0 ]] && return 0
  local t; for t in "${WANT[@]}"; do [[ "$t" == "$1" ]] && return 0; done; return 1
}

mkdir -p outputs/experiments logs

if want 1; then
  echo "### Task 1 -- DCVC-UF image RD (kodim19, qp 0/15/30/45/63)"
  run image-compression "cd /opt/DCVC && python /work/scripts/experiments/exp1_dcvc_image_rd.py" logs/exp1.log
fi

if want 2; then
  echo "### Task 2 -- VTM intra RD (kodim19, QP 22/27/32/37/42)"
  run hybrid-vtm "bash /work/scripts/experiments/exp2_vtm_intra_rd.sh" logs/exp2.log
fi

if want 3; then
  echo "### Task 3 -- DCVC-UF video RD (RaceHorses 64 frames, qp 0/15/30/45/63)"
  run video-compression "cd /opt/DCVC && python /work/scripts/experiments/exp3_dcvc_video_rd.py --frames 64" logs/exp3.log
  # The video overlay compares against VTM, which runs on VTM_FRAMES frames (8 by
  # default). Frames 0-7 and 0-63 are different content, so plot_rd.py needs a
  # DCVC run on the same frame count. Without this, a fresh clone silently drew the
  # overlay with no DCVC curves at all.
  if [[ "${VTM_FRAMES:-8}" != "64" ]]; then
    run video-compression "cd /opt/DCVC && python /work/scripts/experiments/exp3_dcvc_video_rd.py --frames ${VTM_FRAMES:-8} --tag ${VTM_FRAMES:-8}f" logs/exp3_matched.log
  fi
fi

if want 4; then
  echo "### Task 4 -- VTM inter RD (RaceHorses, ${VTM_FRAMES:-8} frames)"
  # Both QP sets: the requested 0/15/30/45/63, and the standard 22..42 that
  # actually overlaps the DCVC rate range. See README_EXPERIMENTS.md.
  run hybrid-vtm "bash /work/scripts/experiments/exp4_vtm_inter_rd.sh --tag std --qps '22 27 32 37 42' --frames ${VTM_FRAMES:-8}" logs/exp4_std.log
  run hybrid-vtm "bash /work/scripts/experiments/exp4_vtm_inter_rd.sh --tag requested --qps '0 15 30 45 63' --frames ${VTM_FRAMES:-8}" logs/exp4_requested.log
fi

if want 5; then
  echo "### Task 5 -- AdaIN across styles + alpha sweep"
  run style-transfer "bash /work/scripts/experiments/exp5_adain_styles.sh" logs/exp5.log
fi

if want 6; then
  echo "### Task 6 -- Real-ESRGAN vs classical resampling"
  run super-resolution "python /work/scripts/experiments/exp6_sr_compare.py" logs/exp6.log

  echo "### Task 6b -- why the fidelity metrics and the eye disagree"
  # Depends on task 6's outputs, so it always follows it.
  run super-resolution "python /work/scripts/experiments/exp6b_perception_distortion.py" logs/exp6b.log
fi

if want 7; then
  echo "### Task 7 -- inference time per project"
  # Dispatches into all five containers itself.
  bash scripts/experiments/run_timing.sh 2>&1 | tee logs/exp7.log
  rc=${PIPESTATUS[0]}; [[ $rc -eq 0 ]] || FAILED+=("logs/exp7.log (exit $rc)")
fi

if want 8; then
  echo "### Task 8 -- real DCVC-UF-Intra bitstreams on CPU"
  run image-compression "cd /opt/DCVC && python /work/scripts/bitstream/uf_bitstream_demo.py" logs/exp8.log
  echo "### Task 8b -- real DCVC-UF video bitstreams on CPU (~8 min)"
  run video-compression "cd /opt/DCVC && python /work/scripts/bitstream/uf_video_bitstream_demo.py" logs/exp8b.log
fi

echo
echo "### Figures"
run image-compression "cd /opt/DCVC && python /work/scripts/experiments/plot_rd.py all" logs/figures_rd.log
run style-transfer "python /work/scripts/experiments/make_figures.py all" logs/figures_sheets.log

echo
echo "figures in results/figures/:"
ls -1 results/figures/ | sed 's|^|  |'

echo
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "FAILED steps:"
  printf '  %s\n' "${FAILED[@]}"
  exit 1
fi
echo "all steps succeeded"
