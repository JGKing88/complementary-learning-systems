#!/bin/bash
# Freeze the current commit as a code snapshot that a Slurm job runs from, so
# later edits to the worktree cannot change a queued or running job.
#
#   scripts/snapshot_code.sh <tag>      # -> $CLS_RUNS/code/<tag>
#
# The snapshot is `git archive HEAD`, which has no .git, so it also gets
# GIT_COMMIT (sha, branch, dirty=false) at its root; run_manifest.py reads that
# when git cannot answer, and every run.json then names the commit it ran.
# Refuses a dirty tree (the snapshot would not match any commit) and an
# existing tag (a running job may be reading it).
set -euo pipefail

tag=${1:?usage: scripts/snapshot_code.sh <tag>}
root=$(git rev-parse --show-toplevel)
dest=${CLS_CODE_ROOT:-/orcd/pool/003/jackking/cls_runs/code}/$tag

if [ -n "$(git -C "$root" status --porcelain --untracked-files=no)" ]; then
  echo "refusing: $root has uncommitted changes to tracked files; commit first" >&2
  exit 1
fi
if [ -e "$dest" ]; then
  echo "refusing: $dest exists (a job may be running from it); pick a new tag" >&2
  exit 1
fi

sha=$(git -C "$root" rev-parse HEAD)
branch=$(git -C "$root" rev-parse --abbrev-ref HEAD)
mkdir -p "$dest"
git -C "$root" archive HEAD | tar -x -C "$dest"
printf '{"sha": "%s", "branch": "%s", "dirty": false}\n' "$sha" "$branch" > "$dest/GIT_COMMIT"
echo "$dest  ($branch @ ${sha:0:7})"
