#!/usr/bin/env bash
set -e
cd /home/swapnil/NEXUS
echo "Starting 1-Hour Autonomous AI Trade Test..."
nohup env PYTHONUNBUFFERED=1 uv run python scripts/run_1h_trade_test.py > 1h_test_run.log 2>&1 &
PID=$!
echo "1-Hour Trade Test started with PID: $PID"
echo $PID > 1h_test.pid
