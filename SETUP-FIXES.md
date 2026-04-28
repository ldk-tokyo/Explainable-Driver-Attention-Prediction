# Post-Install Fixes

> 装完 requirements.txt 之后还要做的修复。这些不在原仓库 README 里, 但跑 `--eval_text` 必需。

## METEOR Java subprocess + paraphrase-en.gz

**症状**: 跑 `train_ds.py ... --eval_text` 时 METEOR 在第 0 个样本就崩, Python 报 `BrokenPipeError`, 系统残留 zombie Java 进程 (`Z+`)。

**根因**: 仓库里 vendored 的 [utils/eval_utils/meteor/meteor.py](utils/eval_utils/meteor/meteor.py) 启动 Java 时传 `-a data/paraphrase-en.gz`, 但仓库 **没有附带** `data/paraphrase-en.gz` (61.8 MB, 太大不入 git)。Java 找不到文件秒退, Python 写 stdin 时管道已断。

**修复** (一次性, 装环境后跑):

```bash
# 1. 装 pycocoevalcap (它 bundle paraphrase-en.gz)
.venv/bin/pip install pycocoevalcap==1.2

# 2. 把 paraphrase-en.gz 复制到项目 vendored meteor 的 data/ 目录
mkdir -p utils/eval_utils/meteor/data
cp .venv/lib/python3.11/site-packages/pycocoevalcap/meteor/data/paraphrase-en.gz \
   utils/eval_utils/meteor/data/
```

**验证**:

```bash
# 文件应是 61813011 bytes
ls -la utils/eval_utils/meteor/data/paraphrase-en.gz

# Java 8 路径(meteor.py:17 硬编码,如果不存在要装 openjdk-8-jdk)
ls /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java
```

如果系统没 Java 8: `sudo apt install -y openjdk-8-jdk` (Ubuntu 系)。

## 不要 git-track paraphrase-en.gz

61.8 MB 超 GitHub 50 MB 警戒线, 也没必要入库 (pycocoevalcap 已经带)。在 `.gitignore` 里加:

```
utils/eval_utils/meteor/data/
```

(目前 `.gitignore` 只有 `.venv/`, 这个 data 目录是 untracked, 不会被误 commit)。

## CUDA / Java 版本

| 工具 | 路径 | 版本 |
| --- | --- | --- |
| CUDA | `/usr/local/cuda-12.8` | 12.8+ (Blackwell sm_120 必需) |
| Java (METEOR) | `/usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java` | 8 (meteor.py 硬编码路径) |
| Java (系统) | `/usr/bin/java` | 21 (无关, 不影响 METEOR) |

## env vars (必备)

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PWD/.venv/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="12.0"
```

`PATH` 里 `.venv/bin` 必须在前面, 否则 deepspeed 起 FusedAdam JIT 时找不到 ninja 会炸 (memory: feedback_deepspeed_path_ninja)。

## 历史记录

- 2026-04-26 01:33-01:43: 上一个 Claude session (`dd3d2295-...`) 第一次跑 `--eval_text` 撞到这个坑, 自己装 pycocoevalcap + 复制 paraphrase-en.gz 修好。本文档把这个修复固化下来, 避免重装环境时重新踩。
