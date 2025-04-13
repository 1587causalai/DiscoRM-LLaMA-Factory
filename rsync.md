# 使用 rsync 同步本地代码与远程服务器


 **使用脚本:**
*   `./sync_disco.sh push`           # 推送到远程 (不删除远程多余文件)
*   `./sync_disco.sh push-delete`    # 推送到远程 (删除远程多余文件)
*   `./sync_disco.sh pull`           # 从远程拉取 (不删除本地多余文件)
*   `./sync_disco.sh pull-delete`    # 从远程拉取 (删除本地多余文件 - **小心!**)
*   `./sync_disco.sh dry-push`       # 预览 push 会做什么 (不删除)
*   `./sync_disco.sh dry-push-delete`# 预览 push 会做什么 (带删除)
*   `./sync_disco.sh dry-pull`       # 预览 pull 会做什么 (不删除)
*   `./sync_disco.sh dry-pull-delete`# 预览 pull 会做什么 (带删除)

## 1. 背景与目标

我们在本地计算机上编写和修改代码，但需要利用远程服务器的强大算力（例如 GPU）来运行、训练或调试，特别是在机器学习和数据科学项目中。

**场景:**

*   **本地代码库:** `/Users/gongqian/DailyLog/projects/Disco-Reward-Model/DiscoRM-LLaMA-Factory/`
*   **远程服务器代码库:** `/root/DiscoRM-LLaMA-Factory/`
*   **远程服务器连接:** `ssh -p 39666 root@ssh.intern-ai.org.cn -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null`
*   **需求:**
    *   将本地修改的代码高效地同步到远程服务器 (Push)。
    *   将远程服务器上可能产生的修改（例如调试时的临时改动或运行结果相关的代码调整）同步回本地 (Pull)。
    *   保持两边代码库（在排除特定文件/目录后）尽可能一致。
    *   操作尽可能简单方便。

**解决方案:** 使用 `rsync` 工具进行文件同步。`rsync` 是一种快速、通用的文件复制工具，它能智能地只传输文件的差异部分，非常适合代码同步。

## 2. 先决条件

1.  **安装 `rsync`:**
    *   macOS 通常自带 `rsync`。可以在终端运行 `rsync --version` 检查。如果未安装或版本过旧，可以通过 Homebrew 安装：`brew install rsync`。
    *   Linux 服务器通常也自带 `rsync`。
2.  **SSH 访问:** 确保你能够通过提供的 SSH 命令成功连接到远程服务器。为了方便起见，强烈建议配置 **SSH 密钥认证**，这样在运行 `rsync` 时就无需每次都输入密码。

## 3. 核心 `rsync` 命令

`rsync` 的基本语法是 `rsync [选项] <源路径> <目标路径>`。

**重要注意事项:**

*   **路径末尾的斜杠 `/`:**
    *   如果**源路径**末尾有 `/`，表示复制该目录的*内容*。
    *   如果**源路径**末尾*没有* `/`，表示复制该目录*本身*及其内容。
    *   对于我们的场景，通常在源路径和目标路径末尾都加上 `/`，表示同步目录内容。
*   **`-e ssh` 选项:** 用于指定通过 SSH 进行远程同步，并可以传递 SSH 的特定参数（如端口号、选项）。

### 3.1. 推送本地代码到远程 (Push)

此命令将本地目录的内容同步到远程目录。

```bash
rsync -avz --progress \
--exclude='.git' \
--exclude='.vscode' \
--exclude='__pycache__' \
--exclude='*.pyc' \
--exclude='venv/' \
--exclude='.venv/' \
--exclude='data/'       # 示例: 排除数据目录
--exclude='logs/'       # 示例: 排除日志目录
--exclude='outputs/'    # 示例: 排除输出/结果目录
--exclude='wandb/'      # 示例: 排除 wandb 日志
--exclude='*.pth'     # 示例: 排除模型权重文件
# --delete             # 可选: 删除远程有但本地没有的文件 (谨慎使用!)
-e 'ssh -p 39666 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null' \
/Users/gongqian/DailyLog/projects/Disco-Reward-Model/DiscoRM-LLaMA-Factory/ \
root@ssh.intern-ai.org.cn:/root/DiscoRM-LLaMA-Factory/
```

**命令选项解释:**

*   `-a` (archive): 归档模式，保留文件权限、时间戳等信息，并递归同步子目录。
*   `-v` (verbose): 显示详细的同步过程信息。
*   `-z` (compress): 传输时压缩数据，节省带宽。
*   `--progress`: 显示每个文件的传输进度。
*   `--exclude='PATTERN'`: 排除匹配指定模式的文件或目录。**这是非常重要的部分，你需要根据你的项目仔细配置：**
    *   `.git`: Git 仓库元数据。
    *   `.vscode`: VS Code 工作区配置。
    *   `__pycache__`, `*.pyc`: Python 编译缓存。
    *   `venv/`, `.venv/`: Python 虚拟环境（通常很大且包含特定于机器的链接）。
    *   `data/`, `logs/`, `outputs/`: 常用于存放大型数据、日志、实验结果，通常不需要或不应该同步。
    *   `wandb/`: Weights & Biases 的日志目录。
    *   `*.pth`, `*.safetensors` 等: 大型模型权重文件。
    *   **务必根据你的实际项目结构调整排除列表！**
*   `--delete` (可选，默认注释掉): **这是一个需要特别注意的选项！** 如果启用，`rsync` 会删除**目标目录**中存在、但**源目录**中不存在的文件（在排除列表之外）。这能确保目标目录是源目录的精确镜像，保持远程环境干净。**但在不确定时，建议先不使用或配合 `-n` (dry-run) 预览效果。**
*   `-e 'ssh ...'`: 指定用于连接的 SSH 命令及参数。
*   **源路径:** 本地代码目录路径 (末尾有 `/`)。
*   **目标路径:** 远程代码目录路径 (末尾有 `/`)。

### 3.2. 从远程拉取代码到本地 (Pull)

此命令将远程目录的内容同步回本地目录。主要是将源和目标对调。

```bash
rsync -avz --progress \
--exclude='.git' \
--exclude='.vscode' \
--exclude='__pycache__' \
--exclude='*.pyc' \
--exclude='venv/' \
--exclude='.venv/' \
--exclude='data/'
--exclude='logs/'
--exclude='outputs/'
--exclude='wandb/'
--exclude='*.pth'
# --delete             # 可选: 删除本地有但远程没有的文件 (更加谨慎使用!)
-e 'ssh -p 39666 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null' \
root@ssh.intern-ai.org.cn:/root/DiscoRM-LLaMA-Factory/ \
/Users/gongqian/DailyLog/projects/Disco-Reward-Model/DiscoRM-LLaMA-Factory/
```

**关于在 Pull 时使用 `--delete`:**

*   在 Pull 操作中使用 `--delete` 需要**格外小心**。它会删除本地存在但远程不存在的文件（非排除项）。如果你本地有一些未提交的更改、临时文件或者不由 Git 管理的文件，它们可能会被删除。
*   通常，Pull 操作只是为了获取远程的更新，不加 `--delete` 会更安全，它只会更新本地已有的文件并下载远程新增的文件。

## 4. 关于 `--delete` 选项的考量

*   **何时使用？**
    *   **Push (本地 -> 远程):** 如果你希望远程服务器上的代码库严格反映本地的状态（删除本地已删除或重命名的文件），并且你确信你的 `--exclude` 列表是完善的，那么可以在 Push 时使用 `--delete`。这有助于保持远程环境整洁。
    *   **Pull (远程 -> 本地):** 除非你确定远程是代码的唯一“真实来源”，并且希望本地完全镜像远程状态，否则**不建议**常规使用。
*   **风险:** 误删文件，尤其是在目标目录（远程或本地）包含重要但未被正确排除的文件时。
*   **建议:**
    *   **始终优先配置好 `--exclude` 列表。**
    *   在启用 `--delete` 前，先使用 **Dry Run** (`-n` 选项) 预览将要执行的操作。
    *   对于 Push 操作，可以考虑使用 `--delete` 以保持远程清洁。
    *   对于 Pull 操作，默认**不使用** `--delete` 通常更安全。

## 5. 简化操作：Shell 脚本

为了避免每次都输入长长的命令，可以创建一个 Shell 脚本来执行同步操作。

1.  **创建脚本文件:** 在你的本地计算机上创建一个文件，例如 `sync_disco.sh` (可以放在项目目录外层或 `~/bin` 等地方)。
2.  **编辑脚本内容:**

    ```bash
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
    ```

3.  **添加执行权限:**
    `chmod +x sync_disco.sh`

4.  **使用脚本:**
    *   `./sync_disco.sh push`           # 推送到远程 (不删除远程多余文件)
    *   `./sync_disco.sh push-delete`    # 推送到远程 (删除远程多余文件)
    *   `./sync_disco.sh pull`           # 从远程拉取 (不删除本地多余文件)
    *   `./sync_disco.sh pull-delete`    # 从远程拉取 (删除本地多余文件 - **小心!**)
    *   `./sync_disco.sh dry-push`       # 预览 push 会做什么 (不删除)
    *   `./sync_disco.sh dry-push-delete`# 预览 push 会做什么 (带删除)
    *   `./sync_disco.sh dry-pull`       # 预览 pull 会做什么 (不删除)
    *   `./sync_disco.sh dry-pull-delete`# 预览 pull 会做什么 (带删除)

## 6. 最佳实践与建议

1.  **SSH 密钥认证:** 配置免密登录极大提升便利性。本地运行 `ssh-keygen` 生成密钥对，然后将公钥 (`~/.ssh/id_rsa.pub` 或其他) 内容追加到远程服务器的 `~/.ssh/authorized_keys` 文件中。
2.  **精确配置 `--exclude`:** 这是 `rsync` 方案安全高效的关键。花时间检查你的项目，确保所有不需要同步的大文件、本地配置、敏感信息、虚拟环境、编译产物、日志和数据集都被排除。
3.  **善用 Dry Run:** 在执行实际的同步操作（尤其是带 `--delete` 的）之前，总是先运行对应的 `dry-*` 命令 (`rsync -n`)，仔细检查 `rsync` 计划执行的文件创建、更新和删除列表，确保符合预期。
4.  **结合 Git:** `rsync` 用于文件同步，**不能替代 Git** 进行版本控制。最佳实践是：
    *   在本地使用 Git 进行代码的版本管理（commit, branch 等）。
    *   在将代码推送到远程服务器运行前，先 `git commit` 保存你的更改。
    *   使用 `rsync` (如 `push` 命令) 将代码同步到服务器。
    *   如果在服务器上做了临时的代码修改并验证有效，记得使用 `rsync` (如 `pull` 命令) 将这些修改同步回本地，然后在本地 `git add` 和 `git commit` 这些改动。
5.  **考虑 VS Code Remote - SSH:** 虽然 `rsync` 方案灵活可控，但 VS Code 的 `Remote - SSH` 扩展提供了另一种更无缝的集成体验。它让你直接在 VS Code 中连接到远程服务器，像编辑本地文件一样编辑服务器上的文件，终端、调试器、扩展都运行在远程，无需手动同步。如果你追求更紧密的集成开发环境，可以尝试此扩展。

## 7. 总结

`rsync` 提供了一个强大而灵活的方式来同步本地和远程服务器之间的代码文件。通过仔细配置选项（尤其是 `--exclude` 和 `--delete`）并结合 Shell 脚本，可以创建一个高效、可控的工作流，满足在本地编辑、远程运行的开发需求。



**使用说明:**

1.  将上面的 Markdown 内容保存到一个 `.md` 文件中（例如 `sync_guide.md`）。
2.  你可以使用任何支持 Markdown 预览的编辑器（如 VS Code、Typora、Obsidian）或在线工具查看格式化后的文档。
3.  确保脚本中的路径、用户名、主机名、端口号以及 `--exclude` 列表都根据你的实际情况进行了修改。

这个文档应该能清晰地解释整个方案，并为你提供具体的命令和脚本。