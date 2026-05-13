#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/cleanup_generated_untracked.sh          # dry-run
  bash scripts/cleanup_generated_untracked.sh --apply  # delete matched untracked generated files

Only untracked files reported by git are considered. Tracked files are never
deleted by this script.
EOF
}

apply=0
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
elif [[ "${1:-}" == "--apply" ]]; then
  apply=1
elif [[ $# -gt 0 ]]; then
  usage >&2
  exit 2
fi

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

matches_generated_path() {
  local path="$1"
  case "$path" in
    outputs/*) return 0 ;;
    saved_models/*) return 0 ;;
    split_ast_mae_runs/*) return 0 ;;
    split_ast_mae_runs*/*) return 0 ;;
    split_ast_local_artifacts/*) return 0 ;;
    split_ast_static_131_features/*) return 0 ;;
    .split_ast_pydeps/*) return 0 ;;
    paper2601_splitmae_runs/*) return 0 ;;
    paper2601_splitmae_runs*/*) return 0 ;;
    paper2601_local_artifacts/*) return 0 ;;
    paper2601_static_131_features/*) return 0 ;;
    .paper2601_pydeps/*) return 0 ;;
    outputs/manual-20260512-paper2601/*) return 0 ;;
    __pycache__/*|*/__pycache__/*) return 0 ;;
    .pytest_cache/*|*/.pytest_cache/*) return 0 ;;
    *.pt|*.pth|*.pptx) return 0 ;;
    split_ast_results_report.html|split_ast_results_report.tsv) return 0 ;;
    split_ast_domain_shift_report.html|split_ast_domain_shift_report.json) return 0 ;;
    split_ast_static_domain_shift.tsv|split_ast_patient_score_shift.tsv) return 0 ;;
    split_ast_stable_static_domain_shift_report.html) return 0 ;;
    split_ast_stable_static_domain_shift_report.json) return 0 ;;
    split_ast_stable_static_domain_shift.tsv) return 0 ;;
    split_ast_stable_static_patient_score_shift.tsv) return 0 ;;
    paper2601_results_report.html|paper2601_results_report.tsv) return 0 ;;
    paper2601_domain_shift_report.html|paper2601_domain_shift_report.json) return 0 ;;
    paper2601_static_domain_shift.tsv|paper2601_patient_score_shift.tsv) return 0 ;;
    paper2601_stable_static_domain_shift_report.html) return 0 ;;
    paper2601_stable_static_domain_shift_report.json) return 0 ;;
    paper2601_stable_static_domain_shift.tsv) return 0 ;;
    paper2601_stable_static_patient_score_shift.tsv) return 0 ;;
  esac
  return 1
}

targets=()
collect_git_paths() {
  while IFS= read -r -d '' path; do
    if matches_generated_path "$path"; then
      targets+=("$path")
    fi
  done
}

collect_git_paths < <(git ls-files --others --exclude-standard -z)
collect_git_paths < <(git ls-files --others --ignored --exclude-standard -z)

if [[ ${#targets[@]} -eq 0 ]]; then
  echo "No matched untracked generated files."
  exit 0
fi

if [[ "$apply" -eq 0 ]]; then
  echo "Dry-run. Matched untracked generated files:"
  printf '  %s\n' "${targets[@]}"
  echo
  echo "Run with --apply to delete them."
  exit 0
fi

printf 'Deleting %d untracked generated files...\n' "${#targets[@]}"
for path in "${targets[@]}"; do
  rm -f -- "$path"
done

find outputs saved_models split_ast_mae_runs split_ast_mae_runs* split_ast_local_artifacts \
  split_ast_static_131_features .split_ast_pydeps paper2601_splitmae_runs paper2601_splitmae_runs* \
  paper2601_local_artifacts paper2601_static_131_features .paper2601_pydeps \
  -depth -type d -empty -delete 2>/dev/null || true

echo "Cleanup complete."
