#!/bin/bash
# Weekly HeatRadar cron — runs every Monday at 6:00am
# Saves a new forecast and verifies the one from 7 days ago.
#
# To install:
#   chmod +x nowcast/cron_weekly.sh
#   crontab -e
#   Add: 0 6 * * 1 /Users/elifkilic/Desktop/ekilic-coder/Heat-Radar/nowcast/cron_weekly.sh

REPO=/Users/elifkilic/Desktop/ekilic-coder/Heat-Radar
PYTHON=/usr/bin/python3
SCRIPT=$REPO/nowcast/src/heatradar_nowcast.py
LOG=$REPO/nowcast/cron.log
RUN_NAME=elif

cd "$REPO"

echo "======================================" >> "$LOG"
echo "$(date): Weekly HeatRadar run starting" >> "$LOG"

# --- Run inference for all 6 MET levels ---
for m in 1 2 3 4 5 6; do
    echo "$(date): Inference MET $m" >> "$LOG"
    $PYTHON "$SCRIPT" \
        --data_dir nowcast/data \
        --lookup_dir shared/ehi/lookup_tables \
        --mode inference \
        --met_level $m \
        --run_name $RUN_NAME >> "$LOG" 2>&1
done

# --- Verify forecasts from 7 days ago for all 6 MET levels ---
LAST_WEEK=$(date -v-7d +%Y%m%d 2>/dev/null || date -d "7 days ago" +%Y%m%d)

for m in 1 2 3 4 5 6; do
    FORECAST_FILE="$REPO/output/met${m}/${RUN_NAME}/forecast_${LAST_WEEK}.json"
    if [ -f "$FORECAST_FILE" ]; then
        echo "$(date): Verifying MET $m forecast from $LAST_WEEK" >> "$LOG"
        $PYTHON "$SCRIPT" \
            --mode verify \
            --met_level $m \
            --run_name $RUN_NAME \
            --forecast_file "$FORECAST_FILE" >> "$LOG" 2>&1
    else
        echo "$(date): No forecast found for MET $m from $LAST_WEEK — skipping verify" >> "$LOG"
    fi
done

echo "$(date): Weekly run complete" >> "$LOG"
