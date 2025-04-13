#!/bin/bash

# --- 配置区 ---
LOCAL_DIR="/Users/gongqian/DailyLog/projects/Disco-Reward-Model/DiscoRM-LLaMA-Factory/"
REMOTE_USER="root"
REMOTE_HOST="ssh.intern-ai.org.cn"
REMOTE_PORT="39666"
REMOTE_DIR="/root/DiscoRM-LLaMA-Factory/"

# SSH 连接选项
SSH_OPTS="-p ${REMOTE_PORT} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
SSH_CMD="ssh ${SSH_OPTS}"

# rsync 基础选项
RSYNC_BASE_OPTS="-avz --progress"

# 排除列表 (根据你的项目调整!)
EXCLUDES=(
  --exclude='.git'
  --exclude='.vscode'
  --exclude='__pycache__'
  --exclude='*.pyc'
  --exclude='venv/'
  --exclude='.venv/'
  --exclude='data/'
  --exclude='logs/'
  --exclude='outputs/'
  --exclude='wandb/'
  --exclude='*.pth'
  # 在这里添加更多需要排除的模式
)

# 远程目标/源 字符串
REMOTE_PATH="${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}"

# --- 功能函数 ---
fn_sync() {
  local mode=$1 # 'push' or 'pull'
  local use_delete=$2 # 'delete' or ''
  local dry_run=$3 # '-n' or ''

  local rsync_opts="${RSYNC_BASE_OPTS}"
  local action_desc=""
  local source_path=""
  local dest_path=""

  if [ "$mode" == "push" ]; then
    action_desc="Pushing local changes to remote"
    source_path="${LOCAL_DIR}"
    dest_path="${REMOTE_PATH}"
  elif [ "$mode" == "pull" ]; then
    action_desc="Pulling remote changes to local"
    source_path="${REMOTE_PATH}"
    dest_path="${LOCAL_DIR}"
  else
    echo "Error: Invalid mode '$mode'"
    exit 1
  fi

  if [ "$use_delete" == "delete" ]; then
    rsync_opts="${rsync_opts} --delete"
    action_desc="${action_desc} (with delete)"
  fi

  if [ "$dry_run" == "-n" ]; then
    rsync_opts="${dry_run} ${rsync_opts}" # Add -n for dry run
    action_desc="DRY RUN: ${action_desc}"
  fi

  echo "🚀 ${action_desc}..."
  echo "rsync ${rsync_opts} ${EXCLUDES[@]} -e \"${SSH_CMD}\" ${source_path} ${dest_path}"

  # Execute the command
  rsync ${rsync_opts} "${EXCLUDES[@]}" -e "${SSH_CMD}" "${source_path}" "${dest_path}"

  local exit_code=$?
  if [ $exit_code -eq 0 ]; then
    echo "✅ Sync complete."
  else
    echo "❌ Sync failed with exit code ${exit_code}."
  fi
  return $exit_code
}

# --- 主逻辑 ---
case "$1" in
  push)
    fn_sync "push" "" "" # Default push without delete
    ;;
  push-delete)
    fn_sync "push" "delete" "" # Push with delete
    ;;
  pull)
    fn_sync "pull" "" "" # Default pull without delete
    ;;
  pull-delete)
    fn_sync "pull" "delete" "" # Pull with delete (use with caution!)
    ;;
  dry-push)
    fn_sync "push" "" "-n" # Dry run push without delete
    ;;
  dry-push-delete)
    fn_sync "push" "delete" "-n" # Dry run push with delete
    ;;
  dry-pull)
    fn_sync "pull" "" "-n" # Dry run pull without delete
    ;;
  dry-pull-delete)
    fn_sync "pull" "delete" "-n" # Dry run pull with delete
    ;;
  *)
    echo "Usage: $0 [push|push-delete|pull|pull-delete|dry-push|dry-push-delete|dry-pull|dry-pull-delete]"
    echo "  push             : Sync local to remote (no delete)"
    echo "  push-delete      : Sync local to remote (deletes extra remote files)"
    echo "  pull             : Sync remote to local (no delete)"
    echo "  pull-delete      : Sync remote to local (deletes extra local files - CAUTION!)"
    echo "  dry-push         : Preview push (no delete)"
    echo "  dry-push-delete  : Preview push (with delete)"
    echo "  dry-pull         : Preview pull (no delete)"
    echo "  dry-pull-delete  : Preview pull (with delete)"
    exit 1
    ;;
esac

exit $?