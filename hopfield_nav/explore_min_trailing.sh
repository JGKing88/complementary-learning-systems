#!/bin/bash
# Trailing-window summary of the explore-min runs.
#
# The 4-env x 16-trial in-training eval is noisy enough that a single final
# reading is not a level: c2s43 reads 0.452 at u1000 with 0.315/0.322/0.387
# immediately before it. This averages the last N evals (default 8 = the last
# 200 updates) so runs are compared on where they SIT, not where their last
# coin landed. Max is printed too, since it is what an early-stopped config
# would actually deliver.
#
# Built on explore_min_status.sh rather than re-parsing the logs, because
# [navigate_uN] appears on both the expl= line and the eval_seconds line and
# a naive tag grep double-counts.

cd "$(dirname "$0")" || exit 1
REPO=${REPO_DIR:-/orcd/home/002/jackking/cls/.claude/worktrees/explore-min-wave2}
N=${N:-8}

REPO_DIR="$REPO" bash "$REPO/hopfield_nav/explore_min_status.sh" 2>/dev/null \
  | awk -v N="$N" '
    $1=="variant" || NF<8 {next}
    $3=="after_navigate" {next}
    {
        v=$1; j=$2; u=$3; c=$4
        key=v" "j
        if (!(key in seen)) { seen[key]=1; order[++n]=key }
        cnt[key]++
        cov[key,cnt[key]]=c
        upd[key,cnt[key]]=u
    }
    END{
        printf "%-7s %-9s %6s %7s %7s %7s %7s %7s\n",
               "variant","job","evals","final","mean"N,"sd"N,"max","@max"
        for(i=1;i<=n;i++){
            key=order[i]; k=cnt[key]
            mx=0; at=""
            for(x=1;x<=k;x++) if(cov[key,x]>mx){mx=cov[key,x]; at=upd[key,x]}
            start=(k>N)?k-N+1:1
            s=0; m=0
            for(x=start;x<=k;x++) s+=cov[key,x]
            w=k-start+1; m=s/w
            vv=0
            for(x=start;x<=k;x++) vv+=(cov[key,x]-m)^2
            sd=(w>1)?sqrt(vv/(w-1)):0
            split(key,p," ")
            printf "%-7s %-9s %6d %7.3f %7.3f %7.3f %7.3f %7s\n",
                   p[1], p[2], k, cov[key,k], m, sd, mx, at
        }
    }'
