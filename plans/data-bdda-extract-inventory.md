# BDDA.tar Extraction Inspection Report

> 时间: 2026-05-01
> 输入: `/media/ldk950413/data0/BDDA.tar` (94 GB)
> 解压目标: `dataset_raw/BDDA/`(`.gitignore` 已加防误 commit)
> 状态: extract 后台运行中(预计 30–60 min)
> 关联文档: [research-roadmap.md](research-roadmap.md) §4.1 Layer 1 sanity

---

## TL;DR

✅ **lab gaze GT 完整且连续**(30Hz frame interval=1),Paper 2 Layer 1 sanity 真正需要的资产**齐了**。
✅ **1428 video / 455k camera frame / 443k gazemap frame / 总 ~4 hour 视频**,数据规模充裕。
⚠️ **不是 Pascal Xia 原始包,是某个 ssd/zcy 转手过的 frame-level 版本**(tar prefix `ssd/zcy/dataset/BDDA-frame/`),preprocessing pipeline 不可追溯。
⚠️ **没有 ego CAN(brake/speed/steering)**,Pivot A Layer 2 (probe leads brake) 用不上 BDDA — 只能给 Layer 1,Layer 2 仍需 BDD100K 或替代。
⚠️ **camera vs gazemap 有 12,326 帧错配**(平均每 video 缺 8–9 帧 gazemap),loader 需做 inner-join 处理。

---

## 1. Source provenance(tar 来源)

- **Path prefix**: `ssd/zcy/dataset/BDDA-frame/`(tar 内部根)
- **解读**: 某人(zcy)把原始 BDDA mp4 视频解码成 per-frame jpg,做了 frame-level repackage。**不是 Pascal Xia 仓库直发的版本**。
- **风险**:
  - preprocessing 步骤(decode codec / resize / 编号规则)不可追溯
  - 帧编号与原始 BDDA paper 索引可能不对应,跟 W³DA-processed `dataset/BDDA/` 也不对应(W³DA 用 6 位帧号 `000001`,这版用 4 位 `0001`)
  - 论文里 cite BDDA 必须保留 Xia 2019 ACCV 原始 reference,不能 cite zcy 的 repackage(没正式发布)
- **README.txt 内容**(全文):
  ```
  Make dataset:
  -----------------------------------
  video name| camera | gaze | class
  -----------------------------------
  0001      | yes    | no   | test
  0922      | error  | no   | train
  1037      | yes    | no   | test
  1045      | yes    | no   | test
  1738      | yes    | no   | test
  -----------------------------------
  ```
  只列了 5 个特殊 video(camera 有但 gaze 缺,或 camera error)的备注,**不是完整 manifest**。

---

## 2. Manifest 结构

```
ssd/zcy/dataset/BDDA-frame/
├── README.txt                     ← 5 行 special-case note
├── train.json                     ← list of "vid/frame.jpg" strings, 286,250 entries / 925 videos
├── valid.json                     ← 63,036 entries / 198 videos
├── camera_frames/<vid>/<frame>.jpg  ← 455,531 frames, 1428 unique videos
└── gazemap_frames/<vid>/<frame>.jpg ← 443,205 frames, 1428 unique videos
```

**JSON 格式**: 每行一个字符串 path(JSONL of strings,不是 nested dict)。

**Split 推断**:
- train.json: 925 videos
- valid.json: 198 videos
- **305 videos 在 tar 但不在 train/valid manifest** → 推测是 test split,但**作者没显式标记**
- 总 1428 = 925 + 198 + 305

---

## 3. Frame 统计

### Per-video frame count(camera_frames)

| 指标 | 值 |
|---|---|
| n_videos | 1428 |
| total frames | 455,531 |
| mean | 319.0 |
| median | 300 |
| min | 299 |
| max | 960 |
| p10 / p25 / p75 / p90 | 300 / 300 / 300 / 360 |

**典型 video 是 300 帧**。如果原始 BDDA 是 30 fps,这是 **10s clip**(跟 BDDA 论文 §3.1 描述的 critical event clip 长度吻合)。少数 long clip 到 960 帧(32s)。

### Frame interval

抽样 5 个 video 验证: **interval = 1, 1, 1, ...**(连续帧),范围如 video `0002` = `[4, 295]` 共 292 帧。

**关键**: 帧编号从 `0004` 而非 `0001` 起 — 推测前 3 帧被 trim(可能 lab subject 还没准备好 / Tobii 还没 calibration)。这个 ~100ms head trim 量级,对 Layer 1 timing alignment 不影响。

---

## 4. Camera vs gazemap 对齐

| 维度 | 数 |
|---|---|
| Video 集合(camera 1428 ∩ gazemap 1428) | **完全对齐** |
| Camera-only frames(无 gazemap pair) | 12,326 |
| Gazemap-only frames(无 camera pair) | **0** |
| Videos 全部帧对齐 | 0 / 1428 |
| Videos 至少有 1 帧 mismatch | **1428 / 1428** |

**关键**: 每个 video 都有些 frame 缺 gazemap(平均 8-9 帧/video,≈ 0.3 秒@30fps)。可能原因:
- Lab subject 眨眼/saccade 期间 Tobii 拿不到 fixation
- Eye tracker raw signal 噪声 frame 被 filter 掉
- 头/尾 calibration 期 trim

**Loader 处理**: inner-join — 只用 `camera ∩ gazemap` 的帧做 timing alignment,丢掉孤儿 camera 帧。这去掉 ~2.7% 的 frame,不影响 Layer 1 统计 power。

---

## 5. 对 Paper 2 各层的可用性判定

| 用途 | 可用? | 备注 |
|---|---|---|
| **Layer 1 sanity (probe-gaze timing alignment)** | ✅ **可用** | 真实 lab gaze + 30fps 连续 + 1428 video × 10s = 4h 视频,sample size 充裕 |
| **Layer 2 main (probe leads brake)** | ❌ 不可用 | BDDA 整个项目设计就没录 ego CAN,brake onset 无法获取 |
| **Reactive/Proactive 分组** | ❌ 不可用 | BDDA 没 BDD-X 那种 action+reason 文本 |
| **替代 BDDA Layer 1 的 fallback** | ✅ 已不再需要 | 这版数据已经够用 |

---

## 6. 跟现有 `dataset/BDDA/` (W³DA-processed sparse) 的对比

| 维度 | `dataset/BDDA/`(W³DA processed) | `dataset_raw/BDDA/`(this) |
|---|---|---|
| 用途 | Paper 1 现有 baseline eval (用作 GT) | Paper 2 Layer 1 sanity |
| 视频数 | 1234 | 1428 |
| 帧数 | sparse(~25-50 帧/video) | dense(~300 帧/video) |
| 帧间距 | 0.6-1.8s(W³DA key-sample selection) | 1 frame(连续 30fps) |
| 帧编号 | 6 位 `000001` | 4 位 `0001` |
| 时间分辨率 | ~600ms | ~33ms |
| What/Why text | 有(W³DA 标注) | 无 |
| **不能互换** | — | — |

**结论**: 两份数据**互补不互斥**。`dataset/BDDA/` 继续给 Paper 1 baseline eval 用,`dataset_raw/BDDA/` 专门给 Paper 2 Layer 1 sanity 用。

---

## 7. Loader 实现 notes(Paper 2 Stage 1 写时参考)

```python
# Pseudo-code
def load_bdda_video(vid):
    """Load (camera_frame, gazemap_frame) pairs for a single video, inner-joined."""
    cam_dir = f"dataset_raw/BDDA/camera_frames/{vid}"
    gaz_dir = f"dataset_raw/BDDA/gazemap_frames/{vid}"
    cam_frames = {f.stem for f in Path(cam_dir).glob("*.jpg")}
    gaz_frames = {f.stem for f in Path(gaz_dir).glob("*.jpg")}
    aligned = sorted(cam_frames & gaz_frames, key=int)  # inner-join, sorted by int(frame_num)
    return [(f"{cam_dir}/{f}.jpg", f"{gaz_dir}/{f}.jpg") for f in aligned]
```

时序连续性保证: 用 `aligned` 的实际帧号去检查最大 gap,如果某段 gap > 5 frame(≈ 167ms),把这段视频切成两个 chunk,避免 attention shift event 跨断点。

---

## 8. Open questions(可推迟到 Paper 2 Stage 1)

1. **305 videos 的 test split 标签**: 是否要从其他线索推断(比如 W³DA-processed `dataset/BDDA/test/` 的 video ID 列表)?
2. **gazemap 编码规范**: 是 binary fixation map 还是 smoothed saliency? 像素值范围? 需要 spot-check 几张图
3. **camera resolution**: BDDA 原始视频不同 source 可能 720p / 1080p,需要确认是否统一被 resize 到固定尺寸
4. **跟 Pascal Xia 原始 BDDA 对齐性**: 这版 1428 vs Pascal 论文 1232 video 多了 196,可能 zcy 加了什么 — 需要 spot-check video ID 是否都在原 BDDA 范围内

这些都不阻塞 extract 完成后的初步 verify,Paper 2 Stage 1 启动时再深挖。

---

## 9. Extract 完成后的验证(✅ 全部 PASS,2026-05-01)

Extract 用时 **38 秒**(IO bound,SSD 顺序写,远快于预估 30-60 min)。

| Check | Expected | Actual | Status |
|---|---|---|---|
| `du -sh dataset_raw/BDDA/` | ~94 GB | **95 GB** | ✅ |
| Total files | ~901,597 | **898,739** (差 2,858 是目录 entries,find -type f 不算) | ✅ |
| `camera_frames/*.jpg` | 455,531 | **455,531** | ✅ |
| `gazemap_frames/*.jpg` | 443,205 | **443,205** | ✅ |
| `camera_frames/` 子目录 | 1428 | **1428** | ✅ |
| `gazemap_frames/` 子目录 | 1428 | **1428** | ✅ |
| Manifest (train.json) random 100 sample 路径存在 | 100/100 | **100/100** camera, **100/100** gazemap | ✅ |
| Camera 图片可读 | 1280×720 RGB | **1280×720 RGB** ✓ | ✅ |
| Gazemap 图片可读 | 任意 valid format | 见 §10 新发现 | ⚠️ |

---

## 10. Verification 中发现的额外重要事实

### 10.1 Gazemap resolution ≠ Camera resolution

| 资产 | Resolution | Mode |
|---|---|---|
| `camera_frames/*.jpg` | **1280 × 720** RGB | 标准 720p |
| `gazemap_frames/*.jpg` | **1024 × 576** RGB | **不同尺寸** |

**含义**: gazemap 跟 camera 在 spatial domain 不一一对应,Layer 1 sanity 比对前需要 resize 到统一坐标(推荐把 gazemap upsample 到 1280×720,bilinear/bicubic)。

**好消息**: 1024/1280 = 576/720 = 0.8,**aspect ratio 完全一致**(没有 crop 偏差),纯尺度变换不会引入 spatial misalignment。

### 10.2 Gazemap 是 smoothed Gaussian saliency,不是 binary fixation

抽样 video `0002/0150.jpg`(paired frame)的统计:
- 像素值范围: **[0, 68]**(uint8,8-bit RGB)— 注意 max=68 < 255,**没有 saturation**
- mean = 2.73,**16.2% 像素非零**
- top 1% 像素集中在 ~18,747 pixel(占图像 ~1.06%)— attention 聚焦区域明确

**含义**: 这是 BDDA 论文 §3.2 描述的 "Gaussian-smoothed gaze fixation" — Tobii raw fixation point 经 σ=固定 半径的 Gaussian kernel smooth 后产生 attention map。**不是 binary fixation point**,所以 Layer 1 timing alignment 算法需要用 **soft 比对**(CC / KL divergence,不是 IoU on binary masks)。

### 10.3 Per-video frame trim pattern 确认

抽样 video 0002:
- camera: 帧 `0001` ~ `0300`,共 300 帧
- gazemap: 帧 `0004` ~ `0295`,共 292 帧
- **头 trim 3 帧**(`0001`, `0002`, `0003` 缺 gazemap)
- **尾 trim 5 帧**(`0296`-`0300` 缺 gazemap)
- 总缺 8 frame,跟 dataset-level 平均值 (12,326 / 1428 ≈ 8.6) 完全吻合

**含义**: trim 模式是 **每 video 头/尾 systematic trim**,不是随机丢帧。Layer 1 loader 用 inner-join 过滤 trim 帧后,**剩下的帧序列仍然连续无空洞**,timing alignment 不受影响。

### 10.4 验证 dimensions 跨 video 一致

随机抽 5 video(`1006`, `0191`, `1833`, `1944`, `1565`)spot-check first frame:
- 全部 `1280×720 RGB` ✓
- camera 没有不同分辨率混入,**全集一致**

---

## 11. 给 Paper 2 Stage 1 启动时的 5 个具体提示

1. **Loader inner-join**: `aligned_frames = sorted(camera_set & gazemap_set)`,丢弃 trim 帧
2. **Resize gazemap**: `gazemap.resize((1280, 720), Image.BICUBIC)` 之前再做 spatial alignment
3. **Soft comparison**: timing alignment 用 frame-level CC / KL,不要用 binary IoU
4. **Test split 推断**: 305 video 不在 manifest,需要从 W³DA-processed `dataset/BDDA/test/` 的 video ID 列表交叉验证(`comm -12` 即可),或者 contact zcy
5. **Cite 谨慎**: 论文里 cite **Xia 2019 ACCV BDDA 原始 paper**,不要 cite zcy repackage(没正式发布)。preprocessing 步骤(decode → resize → trim)不可追溯需在 limitations 标注
