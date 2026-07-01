#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-eiger}"
REMOTE_PATH="/capstor/scratch/cscs/gliu/remote"
DAYS=180
MODE="dry-run"

usage() {
    cat <<'EOF'
Usage:
  scripts/cleanup_eiger_remote_old_jobs.sh [--delete] [--days N] [--path REMOTE_PATH]

Dry-run by default. Lists immediate entries under the remote job directory whose
mtime is older than N days. Pass --delete to remove those entries on Eiger.

Defaults:
  host: eiger, override with REMOTE_HOST
  path: /capstor/scratch/cscs/gliu/remote
  days: 180

Examples:
  scripts/cleanup_eiger_remote_old_jobs.sh
  scripts/cleanup_eiger_remote_old_jobs.sh --days 210
  scripts/cleanup_eiger_remote_old_jobs.sh --delete
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --delete)
            MODE="delete"
            shift
            ;;
        --days)
            DAYS="${2:?missing value for --days}"
            shift 2
            ;;
        --path)
            REMOTE_PATH="${2:?missing value for --path}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ! [[ "$DAYS" =~ ^[0-9]+$ ]] || [[ "$DAYS" -lt 1 ]]; then
    echo "--days must be a positive integer, got: $DAYS" >&2
    exit 2
fi

case "$REMOTE_PATH" in
    /capstor/scratch/cscs/gliu/remote|/capstor/scratch/cscs/gliu/remote/*) ;;
    *)
        echo "Refusing to operate outside /capstor/scratch/cscs/gliu/remote: $REMOTE_PATH" >&2
        exit 2
        ;;
esac

ssh "$REMOTE_HOST" bash -s -- "$REMOTE_PATH" "$DAYS" "$MODE" <<'REMOTE_SCRIPT'
set -euo pipefail

remote_path="$1"
days="$2"
mode="$3"

if [[ ! -d "$remote_path" ]]; then
    echo "Remote path does not exist or is not a directory: $remote_path" >&2
    exit 1
fi

echo "Host: $(hostname)"
echo "Path: $remote_path"
echo "Threshold: older than $days days"
echo "Mode: $mode"
echo

count=0
while IFS= read -r -d '' item; do
    count=$((count + 1))
    stat -c '%y %n' "$item"
done < <(find "$remote_path" -mindepth 1 -maxdepth 1 -mtime +"$days" -print0 | sort -z)

echo
echo "Candidate entries: $count"

if [[ "$count" -eq 0 ]]; then
    exit 0
fi

if [[ "$mode" != "delete" ]]; then
    echo "Dry run only. Re-run with --delete to remove these entries."
    exit 0
fi

echo "Deleting candidates..."
deleted=0
while IFS= read -r -d '' item; do
    rm -rf -- "$item"
    deleted=$((deleted + 1))
done < <(find "$remote_path" -mindepth 1 -maxdepth 1 -mtime +"$days" -print0 | sort -z)

echo "Deleted entries: $deleted"
REMOTE_SCRIPT
