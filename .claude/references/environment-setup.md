# 环境搭建 & 踩坑详录

CLAUDE.md 只给"现在要做什么",本文件给"**为什么这样 / 遇到报错如何判断 / 新机器怎么从头搭**"。

当用户问环境问题、遇到 CUDA/ninja/arch/ModuleNotFoundError 报错、或要在新机器上重建环境时,view 本文件。

---

## 症状 → 诊断速查

| 报错 | 原因 | 修 |
| --- | --- | --- |
| `MissingCUDAException: CUDA_HOME does not exist` | 当前 shell 没 export `CUDA_HOME` | `export CUDA_HOME=/usr/local/cuda-12.8` |
| `Ninja is required to load C++ extensions` | `.venv/bin` 不在 PATH,`ninja` 找不到 | `source .venv/bin/activate` |
| `nvcc fatal: Unsupported gpu architecture 'compute_1.'` | DeepSpeed 0.16.3 arch 生成 bug(见下) | 应用 Patch 1 + `export TORCH_CUDA_ARCH_LIST="12.0"` |
| `ModuleNotFoundError: No module named 'tensorboard'` | requirements.txt 漏写 | `.venv/bin/pip install tensorboard` |
| `ModuleNotFoundError: No module named 'jedi'` | `utils/dataset.py` 的死 import(见 Patch 2) | 应用 Patch 2 删那行 |
| `CUDA SETUP: Required library version not found: libbitsandbytes_cuda128.so` | bitsandbytes 0.41.1 没有 cu128 预编译库 | `pip install -U bitsandbytes`(到 0.49+) |
| `Python.h: 没有那个文件或目录` | 缺 `python3.11-dev` | `sudo apt install python3.11-dev` |
| `E: 无法定位软件包 python3.11` | Ubuntu 24.04 默认源没 python3.11 | 加 deadsnakes PPA(见新机器搭建) |

---

## 新机器从零搭建流程

假设 Ubuntu 24.04 noble + RTX 5090。

### 1. 系统依赖 (apt)

```bash
# Python 3.11(noble 默认源没有,要 PPA)
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev openjdk-8-jdk

# CUDA toolkit 12.8(NVIDIA 官方仓库,Ubuntu 自带的是 12.0.x 不够新)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt update
sudo apt install -y cuda-toolkit-12-8
```

`openjdk-8-jdk` 是 `pycocoevalcap` METEOR/CIDEr 评测依赖。
`python3.11-dev` 提供 `Python.h`,源码编译的包(`pycocotools` 等)需要。

### 2. CUDA 环境变量(加到 `~/.bashrc`)

```bash
echo 'export CUDA_HOME=/usr/local/cuda-12.8' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
nvcc --version   # 应输出 12.8
```

### 3. Python venv

```bash
cd /path/to/Explainable-Driver-Attention-Prediction
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# torch cu128 wheel(Blackwell 最低版本 2.7.0,CLAUDE.md 约束 2.7.1)
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

# 其余依赖(松绑 pycocotools 即可,其他原 pin)
grep -vE "^(torch|torchvision|pycocotools)==" requirements.txt > /tmp/req.txt
echo "pycocotools" >> /tmp/req.txt
pip install -r /tmp/req.txt

# requirements.txt 漏的
pip install tensorboard

# bitsandbytes 0.41.1 无 cu128 库,升级
pip install -U bitsandbytes
```

### 4. 打上游代码 patch

见下方 "上游代码补丁" 一节。

### 5. 验证

```bash
source .venv/bin/activate
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="12.0"

python -c "
import torch, deepspeed, bitsandbytes
print(torch.__version__, torch.cuda.get_device_capability(), torch.cuda.get_device_name(0))
# 期望: 2.7.1+cu128 (12, 0) NVIDIA GeForce RTX 5090
"
```

然后跑 CLAUDE.md §Smoke Test 命令。

---

## 与 requirements.txt 原 pin 的偏离

本地实装和 `requirements.txt` 的差异,都是 CUDA 12.8 + Py3.11 强迫,不是自愿升级:

| 包 | requirements.txt | 本地装 | 为什么 |
| --- | --- | --- | --- |
| `torch` | `2.1.2` | **`2.7.1+cu128`** | Blackwell sm_120 需要 ≥ 2.7 + cu128 wheel |
| `torchvision` | `0.16.2` | **`0.22.1+cu128`** | 匹配 torch 2.7.1 |
| `pycocotools` | `2.0.6` | **(unpinned)** | 2.0.6 无 Py3.11 wheel,源码编译有工具链坑 |
| `bitsandbytes` | `0.41.1` | **`0.49.2`** | 0.41.1 无 `libbitsandbytes_cuda128.so` |
| `tensorboard` | **(漏列)** | 最新 | `train_ds.py:18` 需要 |

其他包(`transformers==4.31.0`, `peft==0.4.0`, `deepspeed==0.16.3`, `numpy==1.24.2`, `Pillow==9.4.0`, `opencv==4.8.0.74`, `scipy==1.11.2` 等)**全部按原 pin**,复现性基本保住。

---

## 上游代码补丁

这些补丁不在 upstream 里,venv 或 deepspeed 重装后要重新打。

### Patch 1: DeepSpeed 0.16.3 Blackwell arch bug

**定位**:`.venv/lib/python3.11/site-packages/deepspeed/ops/op_builder/builder.py` 约 L652-L660

**bug**:硬编码 `num = cc[0] + cc[2]` 假设 compute capability 主版本是单位数。对 sm_120 (cc="12.0"):
- `cc[0]="1"`, `cc[2]="."` → `num="1."` → 生成畸形 `-gencode=arch=compute_1.,code=sm_1.`
- `nvcc fatal: Unsupported gpu architecture 'compute_1.'` 编译 fused_adam 失败

同一处 L658 `if int(cc[0]) <= 7` 对 "12.0" 算出 `1 <= 7`,会**错误地禁用 bf16**(Blackwell 原生支持 bf16)。

**一次性 patch 脚本**:

```bash
python - <<'PY'
p = '.venv/lib/python3.11/site-packages/deepspeed/ops/op_builder/builder.py'
s = open(p).read()
s = s.replace("num = cc[0] + cc[2]",
              "cc_clean = cc.split('+')[0]  # strip +PTX\n            num = cc_clean.replace('.', '')")
s = s.replace("if int(cc[0]) <= 7:",
              "if int(cc_clean.split('.')[0]) <= 7:")
open(p, 'w').write(s)
print('patched', p)
PY
rm -rf ~/.cache/torch_extensions/py311_cu128/fused_adam   # 清 JIT 旧缓存
```

即使打了 patch,也**还需 `export TORCH_CUDA_ARCH_LIST="12.0"`**,否则 DeepSpeed 的 cross-compile 路径会多叠加默认 arch 集合,仍可能触发畸形生成。

### Patch 2: `utils/dataset.py` 死 jedi import

`utils/dataset.py:10` 原本有 `from jedi.api.helpers import infer`。搜遍整个文件,`infer` 符号**从未被调用**,只有同名变量 `inferences` / `inference`。是 IDE 自动补全泄漏的僵尸代码(作者按 `inference` + Tab 自动 import 了,然后代码被 refactor 了但 import 没删)。

装 `jedi` 只是掩盖问题,**删掉那行是正解**。如果从 upstream 合并代码时这行回来,再删一次。

### Patch 2b: `utils/dataset.py` collate_fn 漏截 `targets_what` / `targets_why`

`utils/dataset.py` 的 `collate_fn` 里,对话超过 `tokenizer.model_max_length - 255` 时会截断 `input_ids`/`targets`/`attention_masks`,**但漏了 `targets_what` 和 `targets_why`**。这两个 target 是后加入的(llava_arch.py 里能看到 `# added` 注释),作者加它们时忘了同步更新截断。

**症状**:训练前期不炸,跑几百步后 `llava_arch.py:160` 突然触发:
```
AssertionError: cur_labels_what.shape == cur_input_ids.shape
```
在某个 prompt 特别长的样本上(W3DA 数据集里 DADA 和 LBW 偶尔有超长描述,BDDA/DReyeVE 较短所以前期不触发)。

**fix**(在 `utils/dataset.py` 约 L157-L160 的 `if input_ids.shape[1] > truncate_len:` 块里加两行):

```python
if input_ids.shape[1] > truncate_len:
    input_ids = input_ids[:, :truncate_len]
    targets = targets[:, :truncate_len]
    targets_what = targets_what[:, :truncate_len]   # ← 新增
    targets_why = targets_why[:, :truncate_len]     # ← 新增
    attention_masks = attention_masks[:, :truncate_len]
```

**注意**:这个 bug 只有在**混合 4 数据集训练**时才容易撞到(DADA/LBW 的某些样本触发)。纯 BDDA smoke test 发现不了。

### Patch 3: `requirements.txt` 漏 tensorboard

`train_ds.py:18` 的 `from torch.utils.tensorboard import SummaryWriter` 需要 `tensorboard` 包(PyTorch 的 tensorboard 封装只是 shim),但原 `requirements.txt` 没列。

```bash
.venv/bin/pip install tensorboard
```

---

## 关于 `.venv/bin` 和 ninja 的细节

DeepSpeed 起 deepspeed 子进程时,虽然自己 `Popen(env=os.environ.copy())` 会继承 PATH,但前提是**当前 shell 的 PATH 已经有 `.venv/bin`**。所以:

- ✅ `source .venv/bin/activate && deepspeed ...` 能跑
- ❌ `.venv/bin/deepspeed ...` 不能跑(deepspeed 自己能启动,但它子进程里 `python3.11 train_ds.py` 调用 `ninja` 时走 PATH 找不到)

CLAUDE.md 的 smoke test 命令已经在前面 `source` 了,没问题。

## 关于 DeepSpeed 的 env var 白名单

看 `deepspeed/launcher/runner.py:37`:
```python
EXPORT_ENVS = ['MLFLOW', 'PYTHON', 'MV2', 'UCX']
```

DeepSpeed 的 **multi-node** 路径(PDSH 等)只透传前缀匹配这些的 env var。但**单节点路径**(我们用的 `--num_gpus=1`)直接 `Popen(env=os.environ.copy())`,全量继承。

所以单节点下 `TORCH_CUDA_ARCH_LIST=12.0` 是能传到子进程的(经诊断确认)。多节点时才需要用 `./.deepspeed_env` 文件。
