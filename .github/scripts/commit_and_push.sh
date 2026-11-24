#!/usr/bin/env bash
# Usage: commit_and_push.sh "<commit message>" <path>...
# Commits the given paths (if changed) and pushes, rebasing on top of any
# concurrent push. Retries a few times; exits non-zero if it cannot push.
set -euo pipefail
msg="$1"; shift
git config user.name "github-actions[bot]"
git config user.email "github-actions[bot]@users.noreply.github.com"
git add -A -- "$@"
if git diff --staged --quiet; then
  echo "No changes to commit"
  exit 0
fi
git commit -m "$msg"
for attempt in 1 2 3 4 5; do
  if git pull --rebase origin "${GITHUB_REF_NAME:-main}" && git push origin "HEAD:${GITHUB_REF_NAME:-main}"; then
    echo "Pushed on attempt $attempt"
    exit 0
  fi
  git rebase --abort 2>/dev/null || true
  sleep $((attempt * 15))
done
echo "::error::Could not push after 5 attempts"
exit 1
