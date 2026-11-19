#!/usr/bin/env bash
# Run a resumable, research-only NEXUS improvement cycle without polling it.
# It never submits broker orders.  A final summary is written to reports/.
set -uo pipefail

cd "$(dirname "$0")/.."

min_payout="${NEXUS_MIN_PAYOUT:-80}"
# Fast screening defaults: the five highest-payout eligible markets, compact
# indicators, two chronological folds.  Promotion gates are unchanged.
search_depth="${NEXUS_SEARCH_DEPTH:-rapid}"
indicator_scope="${NEXUS_INDICATOR_SCOPE:-core}"
folds="${NEXUS_WALK_FORWARD_FOLDS:-2}"
market_limit="${NEXUS_RESEARCH_LIMIT:-5}"
max_attempts="${NEXUS_RESEARCH_MAX_ATTEMPTS:-3}"
retry_delay="${NEXUS_RESEARCH_RETRY_DELAY_SECONDS:-120}"
log_path="${NEXUS_RESEARCH_LOG:-reports/unattended_research.log}"
summary_path="${NEXUS_RESEARCH_SUPERVISOR_REPORT:-reports/unattended_research_summary.json}"
# A changed evaluator must never silently reuse a prior report/checkpoint.
force_retrain="${NEXUS_FORCE_RETRAIN:-1}"

mkdir -p reports
: > "$log_path"

attempt=1
attempts_completed=0
exit_code=1
retry_errors_only=0
while [ "$attempt" -le "$max_attempts" ]; do
  echo "[$(date -u +%FT%TZ)] research attempt $attempt/$max_attempts" | tee -a "$log_path"
  if [ "$retry_errors_only" -eq 1 ]; then
    NEXUS_FORCE_RETRAIN="$force_retrain" NEXUS_RESEARCH_LIMIT="$market_limit" NEXUS_RESEARCH_ONLY_ERRORS=1 \
      uv run python scripts/run_robust_research.py \
        --min-payout "$min_payout" --search-depth "$search_depth" \
        --indicator-scope "$indicator_scope" --folds "$folds" 2>&1 | tee -a "$log_path"
  else
    NEXUS_FORCE_RETRAIN="$force_retrain" NEXUS_RESEARCH_LIMIT="$market_limit" uv run python scripts/run_robust_research.py \
      --min-payout "$min_payout" --search-depth "$search_depth" \
      --indicator-scope "$indicator_scope" --folds "$folds" 2>&1 | tee -a "$log_path"
  fi
  exit_code=${PIPESTATUS[0]}
  attempts_completed="$attempt"

  auth_errors="$(uv run python - <<'PY'
import json
from pathlib import Path
p = Path("reports/robust_market_research.json")
if not p.exists():
    print(0)
else:
    report = json.loads(p.read_text(encoding="utf-8"))
    print(sum(item.get("error_type") == "authentication" for item in report.get("markets", [])))
PY
)"
  if [ "$exit_code" -eq 0 ] && [ "$auth_errors" -eq 0 ]; then
    break
  fi
  # A startup/auth failure means no fresh catalog was evaluated.  Its next
  # attempt must run the whole catalog, not silently use stale report entries.
  retry_errors_only=0
  if [ "$exit_code" -eq 0 ]; then
    retry_errors_only=1
  fi
  if [ "$attempt" -lt "$max_attempts" ]; then
    echo "[$(date -u +%FT%TZ)] retrying $auth_errors authentication failures after ${retry_delay}s" | tee -a "$log_path"
    sleep "$retry_delay"
  fi
  attempt=$((attempt + 1))
done

uv run python - "$summary_path" "$attempts_completed" "$exit_code" <<'PY'
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

out = Path(sys.argv[1])
attempts = int(sys.argv[2])
exit_code = int(sys.argv[3])
report_path = Path("reports/robust_market_research.json")
report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {"markets": []}
markets = report.get("markets", [])
complete = [item for item in markets if item.get("status") == "complete"]
errors = [item for item in markets if item.get("status") == "error"]
promoted = [item.get("asset") for item in complete if item.get("promotion_status") == "promote"]
payload = {
    "finished_at": datetime.now(timezone.utc).isoformat(),
    "mode": "research_only",
    "orders_placed": 0,
    "attempts": attempts,
    "last_command_exit_code": exit_code,
    "completed_markets": len(complete),
    "promoted_markets": promoted,
    "error_counts": Counter(str(item.get("error_type", "unknown")) for item in errors),
    "top_rejection_reasons": Counter(
        reason for item in complete for reason in item.get("promotion_rejection_reasons", [])
    ).most_common(),
    "next_action": "run evidence-gated demo only for promoted markets" if promoted else "no demo orders; improve or gather more held-out evidence",
    "research_report": str(report_path),
}
out.write_text(json.dumps(payload, indent=2, default=list), encoding="utf-8")
print(json.dumps(payload, indent=2, default=list))
PY

echo "Finished. Log: $log_path | Summary: $summary_path"
exit "$exit_code"
