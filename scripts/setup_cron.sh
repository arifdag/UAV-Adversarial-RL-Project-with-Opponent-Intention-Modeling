#!/bin/bash
"""
Setup Cron Job for Weekly UAV League Tournaments

This script sets up a cron job to run tournaments weekly
"""

# Get the absolute path of the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_ENV="${PROJECT_DIR}/.venv/bin/python"
SCRIPT_PATH="${PROJECT_DIR}/scripts/weekly_tournament.py"

echo "Setting up weekly tournament cron job..."
echo "Project directory: $PROJECT_DIR"
echo "Python environment: $PYTHON_ENV"
echo "Script path: $SCRIPT_PATH"

# Check if virtual environment exists
if [ ! -f "$PYTHON_ENV" ]; then
    echo "Warning: Virtual environment not found at $PYTHON_ENV"
    echo "Using system python instead"
    PYTHON_ENV="python3"
fi

# Check if script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Tournament script not found at $SCRIPT_PATH"
    exit 1
fi

# Create cron job entry
# Run every Sunday at 2 AM
CRON_ENTRY="0 2 * * 0 cd $PROJECT_DIR && $PYTHON_ENV $SCRIPT_PATH >> $PROJECT_DIR/league/logs/cron.log 2>&1"

echo "Cron job entry:"
echo "$CRON_ENTRY"

# Add to crontab
(crontab -l 2>/dev/null; echo "$CRON_ENTRY") | crontab -

if [ $? -eq 0 ]; then
    echo "✅ Cron job successfully added!"
    echo "Tournament will run every Sunday at 2:00 AM"
    echo ""
    echo "To view current cron jobs:"
    echo "  crontab -l"
    echo ""
    echo "To remove this cron job:"
    echo "  crontab -e"
    echo "  (then delete the line containing 'weekly_tournament.py')"
    echo ""
    echo "Logs will be saved to: $PROJECT_DIR/league/logs/"
else
    echo "❌ Failed to add cron job"
    exit 1
fi 