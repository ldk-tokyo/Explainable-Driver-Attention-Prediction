---
name: w3da-dataset
description: W³DA 及 4 个子数据集 (BDDA/DR(eye)VE/LBW/DADA) 的目录结构、JSON 标注格式、`HybridDataset` 采样机制、接入自建数据的流程。涉及数据加载、帧目录、gazemap、`train_sample_rates`、新增子数据集时用。数据相关的 `FileNotFoundError` / `KeyError` 也用。
---

# W³DA 数据集工作指南

## 一、顶层目录结构 (必须严格遵守)

```
dataset/
├── BDDA/
│   ├── training/
│   │   ├── 0001/
│   │   │   ├── raw_frames/
│   │   │   │   ├── 000001.jpg
│   │   │   │   ├── 000002.jpg
│   │   │   │   └── ...
│   │   │   ├── gazemap_frames/
│   │   │   │   ├── 000001.jpg      # 8-bit 灰度,GT 热力图
│   │   │   │   └── ...
│   │   │   ├── 000001.json         # 帧级 What/Why 标注
│   │   │   ├── 000002.json
│   │   │   └── ...
│   │   ├── 0002/
│   │   └── ...
│   ├── validation/
│   └── test/
├── DReyeVE/  (同构)
├── LBW/      (同构)
└── DADA/     (同构,但事故场景)
```

**每个 split 的视频目录名是数字(`0001`, `0002`...)**,`raw_frames` 和 `gazemap_frames` 下的帧号一一对应。每帧有一个同名 `.json` 存文本标注。

## 二、JSON 标注格式 (必须搞清楚才能做文本评测)

典型内容(以 README 和 `sep_what_and_why` 逻辑反推):

```json
{
  "conversations": [
    {
      "from": "human",
      "value": "<image>\n描述驾驶员在该场景中应该关注的内容..."
    },
    {
      "from": "gpt",
      "value": "1. Where: [ATTN]\n2. What: a pedestrian is crossing the road from the left, a red sedan ahead is braking.\n3. Reason: the pedestrian is entering the vehicle's path, requiring immediate attention to avoid collision."
    }
  ]
}
```

关键:
- **`[ATTN]` token 出现的位置**就是热力图解码器"看"的位置 ("Where" 部分)
- **"3. Reason" 是 What/Why 分界线**(偶尔是 "- Reason" / "Reason" / "3.",见 `sep_what_and_why`)
- 不同子数据集可能措辞略有不同,但结构必须满足 `sep_what_and_why` 能拆分

读一个样例的快速命令:
```bash
ls dataset/BDDA/training/ | head -1 | xargs -I {} cat dataset/BDDA/training/{}/000001.json | python -m json.tool
```

## 三、4 个子数据集的性格差异

| 子数据集 | 场景 | 原始帧率 | 注意点 |
| --- | --- | --- | --- |
| **BDDA** | Safety-critical driving (突发情况多) | 10 FPS | 热力图较集中,事件密度高 |
| **DR(eye)VE** | 日常巡航 (高速/市区) | 25 FPS | 注视稳定,分布广,样本量大 |
| **LBW** | Look Both Ways (需左右观察的路口) | 30 FPS | 样本少,但标注精细 |
| **DADA-2000** | 交通事故场景 | 30 FPS | 事故瞬间前后,注视非常集中 |

**原论文默认采样率**: `train_sample_rates="8,5,2,7"` — 注意数据多少不等于采样比高,作者显然人为拔高了 BDDA 和 DADA(事件密集场景)的权重。做改动实验时如果你只关心某类场景,直接调这个比例。

## 四、`HybridDataset` 的采样逻辑

在 `utils/dataset.py` 里,`HybridDataset` 不是把 4 个数据集串起来后 shuffle,而是:
1. 每个 epoch 目标样本数 = `samples_per_epoch = batch_size × grad_accumulation_steps × steps_per_epoch × world_size`
2. 对每个样本位置,按 `sample_rate` 的归一化概率随机选一个子数据集
3. 在选中的子数据集里随机抽一条

**后果**: 数据"看过一遍"的概念不明确 —— `steps_per_epoch=500` 就是 500 × (effective batch) 个样本,跟真实 epoch 不等。**命名上还叫 epoch 只是习惯**。

## 五、`__getitem__` 产出的字典字段

读懂这些你才能看懂训练代码:

| 字段 | shape / type | 来源 |
| --- | --- | --- |
| `images_clip` | `[3, 224, 224]` | CLIP 预处理后 |
| `images` | `[3, 1024, 1024]` | 为 attn_decoder 保留的高分辨率图 |
| `input_ids` | `[seq_len]` | tokenize 后 (含 `<image>` 占位和 `[ATTN]`) |
| `labels` | `[seq_len]` | CE loss 的 target,prompt 部分为 -100 |
| `masks_list` | `List[Tensor]`, 每个 `[256, 256]` | GT 热力图(`map_size=256`) |
| `conversation_list` | `List[str]` | 原始文本,评测时用 |
| `questions_list` | `List[str]` | 问题 |
| `answers_list` | `List[str]` | 答案(What + Why) |
| `image_paths` | `List[str]` | 原始帧绝对路径,评测时定位样本 |

## 六、接入新数据的标准流程

假设你要加一个 `MyCustomDriveData`:

1. **符合目录约定**: 照抄 BDDA 的结构,video 目录 / raw_frames / gazemap_frames / per-frame JSON
2. **在 `utils/dataset.py` 里新增 class**:
   ```python
   class MyCustomDriveData(Dataset):
       def __init__(self, base_dir, split, ...):
           self.samples = self._scan(base_dir, split)
       def __getitem__(self, idx):
           # 返回同六(五)的字段
           return sample_dict
   ```
3. **注册到 `HybridDataset.__init__`**:
   在 dataset 字符串解析处加一个分支,支持 `--dataset="MyCustomDriveData||BDDA"` 这种用法
4. **补齐采样率**: `--train_sample_rates` 长度要对得上

**GT 热力图的生成**:
- 如果你只有 gaze 点坐标(x, y),用高斯模糊生成: `sigma ≈ 图像宽的 2.5%`
- 论文里 gazemap 都是 `uint8 灰度图`,`cv2.imread(..., 0)` 读进来后 normalize 到 `[0, 1]`
- 保存时 `cv2.imwrite(path, (heatmap*255).astype(np.uint8))`

## 七、数据异常排查决策树

| 症状 | 检查 |
| --- | --- |
| `FileNotFoundError: ...000123.jpg` | 检查帧号是否有前导零(`000123` vs `123`),和 JSON 里存的是否一致 |
| DataLoader 卡在 worker 启动 | `workers=4` 配合 bf16 内存够不够;Linux 共享内存 `/dev/shm` 够不够 |
| 某个子数据集样本量为 0 | `HybridDataset` 里的路径扫描是否跑通;该 split 目录是否非空 |
| `labels` 全是 -100 | prompt 模板里 `ASSISTANT:` 之后的内容没有被标成训练目标,检查 `collate_fn` |
| 热力图可视化一片黑 | gazemap_frames 里的图是 `uint8 0~255`,代码可能期望 `float 0~1`,检查 `HybridDataset.__getitem__` 的 normalize |
| 文本指标全是 0 | 对比 `pred_text` 和 `gt_text`,看是不是 `sep_what_and_why` 拆分错了(新 prompt 模板的分隔符变了) |

## 八、下载数据的正确姿势

```bash
# 方式一: HF CLI (推荐,支持断点续传)
huggingface-cli download JYT4chenxiyuxi/W3DA \
  --repo-type dataset \
  --local-dir ./dataset \
  --local-dir-use-symlinks False

# 方式二: 用 snapshot_download (Python)
from huggingface_hub import snapshot_download
snapshot_download(repo_id="JYT4chenxiyuxi/W3DA", repo_type="dataset",
                  local_dir="./dataset")
```

如果磁盘紧张,可以只下载感兴趣的子数据集:
```bash
huggingface-cli download JYT4chenxiyuxi/W3DA --repo-type dataset \
  --local-dir ./dataset --include "BDDA/*"
```

## 九、别做的事

- **不要**把 raw_frames 和 gazemap_frames 放一起,代码会 walk 目录找帧号
- **不要**在 split 目录下放 `.DS_Store` 或其他杂文件,`HybridDataset` 扫描时会踩
- **不要**手动改 `.json` 里的结构而不同步 `sep_what_and_why`(第七节提过)
- **不要**用软链接跨数据盘 —— `num_workers > 0` 时某些文件系统会出奇怪的 I/O 阻塞
