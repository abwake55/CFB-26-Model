#!/usr/bin/env bash
# Opens (or comments on) one GitHub issue labelled automation-failure so a
# broken scheduled job is visible instead of silently failing for weeks.
set -uo pipefail
run_url="${GITHUB_SERVER_URL}/${GITHUB_REPOSITORY}/actions/runs/${GITHUB_RUN_ID}"
title="Automation failing: ${GITHUB_WORKFLOW}"
body="Workflow **${GITHUB_WORKFLOW}** failed (event: ${GITHUB_EVENT_NAME}) at $(date -u +'%Y-%m-%d %H:%M UTC').

Run: ${run_url}"
gh label create automation-failure --color B60205 --description "Scheduled job failed" 2>/dev/null || true
existing=$(gh issue list --label automation-failure --state open --search "in:title \"${title}\"" --json number --jq '.[0].number' 2>/dev/null || true)
if [ -n "${existing}" ]; then
  gh issue comment "${existing}" --body "${body}"
else
  gh issue create --title "${title}" --label automation-failure --body "${body}"
fi
