#!/usr/bin/env bash
# Bounded autonomous NEXUS cycle: research -> promotion gate -> PRACTICE demo.
# Never places a real-money order.  A broker-confirmed Quotex PRACTICE order is
# possible only after the current research report promotes a market.
set -euo pipefail

if [ "${1:-}" != "--duration-hours" ] || [ -z "${2:-}" ]; then
  echo "Usage: bash scripts/run_autonomous_demo_loop.sh --duration-hours HOURS"
  exit 2
fi

duration_hours="$2"
if ! [[ "$duration_hours" =~ ^[0-9]+([.][0-9]+)?$ ]] || [ "$duration_hours" = "0" ]; then
  echo "HOURS must be a positive number"
  exit 2
fi

cd "$(dirname "$0")/.."
mkdir -p reports
log_path="reports/autonomous_demo_loop.log"
summary_path="reports/autonomous_demo_loop_summary.json"
research_interval_seconds="${NEXUS_AUTONOMOUS_RESEARCH_INTERVAL_SECONDS:-900}"
practice_window_minutes="${NEXUS_AUTONOMOUS_PRACTICE_WINDOW_MINUTES:-30}"
start_epoch="$(date +%s)"
deadline_epoch="$(awk -v start="$start_epoch" -v hours="$duration_hours" 'BEGIN { print int(start + hours * 3600) }')"
cycles=0
research_runs=0
demo_runs=0

echo "[$(date -u +%FT%TZ)] starting bounded autonomous PRACTICE cycle for ${duration_hours}h" | tee "$log_path"
while [ "$(date +%s)" -lt "$deadline_epoch" ]; do
  cycles=$((cycles + 1))
  research_runs=$((research_runs + 1))
  echo "[$(date -u +%FT%TZ)] cycle $cycles: research screen" | tee -a "$log_path"
  bash scripts/run_unattended_research.sh 2>&1 | tee -a "$log_path" || true

  promoted="$(uv run python - <<'PY'
import json
from pathlib import Path
p = Path("reports/robust_market_research.json")
if not p.exists():
    print("")
else:
    report = json.loads(p.read_text(encoding="utf-8"))
    print(",".join(str(item.get("asset")) for item in report.get("markets", []) if item.get("promotion_status") == "promote"))
PY
)"
  now_epoch="$(date +%s)"
  remaining_seconds=$((deadline_epoch - now_epoch))
  if [ "$remaining_seconds" -le 0 ]; then
    break
  fi

  if [ -n "$promoted" ]; then
    demo_runs=$((demo_runs + 1))
    remaining_minutes=$((practice_window_minutes < (remaining_seconds + 59) / 60 ? practice_window_minutes : (remaining_seconds + 59) / 60))
    echo "[$(date -u +%FT%TZ)] promoted: $promoted; starting real PRACTICE demo for ${remaining_minutes}m" | tee -a "$log_path"
    # The demo runner rejects NEXUS_FORCE_SIM and requires broker order IDs
    # plus verified settlement before an outcome can update learning.
    uv run python scripts/run_1h_trade_test.py --duration-minutes "$remaining_minutes" 2>&1 | tee -a "$log_path" || true
  else
    demo_runs=$((demo_runs + 1))
    exploration_minutes=$((practice_window_minutes < (remaining_seconds + 59) / 60 ? practice_window_minutes : (remaining_seconds + 59) / 60))
    echo "[$(date -u +%FT%TZ)] no promotion; starting real PRACTICE exploration for ${exploration_minutes}m" | tee -a "$log_path"
    uv run python scripts/run_1h_trade_test.py --duration-minutes "$exploration_minutes" --practice-exploration 2>&1 | tee -a "$log_path" || true
  fi

  remaining_seconds=$((deadline_epoch - $(date +%s)))
  if [ "$remaining_seconds" -le 0 ]; then
    break
  fi
  sleep_seconds=$((research_interval_seconds < remaining_seconds ? research_interval_seconds : remaining_seconds))
  echo "[$(date -u +%FT%TZ)] no promoted model; next research screen in ${sleep_seconds}s" | tee -a "$log_path"
  sleep "$sleep_seconds"
done

uv run python - "$summary_path" "$duration_hours" "$cycles" "$research_runs" "$demo_runs" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

out = Path(sys.argv[1])
payload = {
    "finished_at": datetime.now(timezone.utc).isoformat(),
    "duration_hours": float(sys.argv[2]),
    "cycles": int(sys.argv[3]),
    "research_runs": int(sys.argv[4]),
    "demo_runs": int(sys.argv[5]),
    "execution_mode": "Quotex PRACTICE only; simulation refused",
    "promotion_required": True,
    "research_orders_placed": 0,
    "log": "reports/autonomous_demo_loop.log",
}
out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2))
PY
