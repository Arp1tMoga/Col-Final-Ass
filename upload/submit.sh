#!/usr/bin/env bash
# Package submission: creates a tarball with code + metadata
set -euo pipefail
TEAM=${1:-team_unknown}
OUTPUT=submit_${TEAM}_$(date +%Y%m%d_%H%M%S).tar.gz

# required files
tar -czf ${OUTPUT} README.md baseline optimized Makefile run.sh submit.sh scorer.py report_template.md

echo "Created submission package: ${OUTPUT}"

echo "Please upload ${OUTPUT} to the course server or submit link in the leaderboard sheet."