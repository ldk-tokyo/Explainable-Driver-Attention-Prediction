# Pilot Feasibility Note — DR(eye)VE assets in this repo

> 目的: 检查当前 `dataset/DReyeVE/` 是否够做 attention-behavior lag pilot。
> 结论: **不够**。当前资产是 W³DA processed sparse subset,缺三样关键东西。
> 行动: 见 §3 三选一,等用户拍板。

---

## 1. 现状清点

**`dataset/DReyeVE/{train,test}/<vid>/`** 每个视频目录下有 4 类资产:

| 资产 | 形态 | 用途 | pilot 可用? |
|---|---|---|---|
| `raw_frames/{N}.jpg` | sparse single frames | RGB | ✗ 抽稀 |
| `gazemap_frames/{N}.png` | sparse fixation map | gaze GT | ✗ 抽稀 |
| `heatmap_frames/{N}.jpg` | smoothed saliency overlay | viz | ✗ 抽稀 |
| `{N}.json` | What/Why text + speed-in-prompt | LLada 监督 | 速度仅以字符串嵌 prompt |

**Sparse 程度**:
- test 共 37 video,每 video 约 150–350 帧
- 帧编号范围 1 → ~7000+ (说明原始视频 25fps × 5min ≈ 7500 帧)
- 帧间距分布: **60% 是 15 帧(0.6s)** / 30% 是 30 帧(1.2s)/ 10% 是 45 帧(1.8s)
- 这是 W³DA `attention-aware key sample selection` 的产物(论文 §3.1):用 KL divergence + CLIP semantic similarity 抽稀过的"高信息帧"

**目录尺寸**: `dataset/DReyeVE/` total **16 GB**(W³DA sparse subset)。原始 DR(eye)VE 公开版 ~80 GB raw video + ~10 GB metadata。

---

## 2. Pilot 需要但当前缺失的 3 样东西

| 资产 | 当前状态 | Pilot 需求 | gap |
|---|---|---|---|
| **连续 25fps gaze trace** | sparse 0.6–1.8s 间隔 | brake onset 前 3s 窗口的 ~75 连续帧 | 当前 3s 窗内只 2–5 帧,无法做 lag |
| **Ego CAN trace** (brake / speed / steering 时间序列) | 无 — 速度仅以字符串嵌 prompt 文本 ("traveling at 52 km/h") | brake onset detection 必需(speed 导数 < -2.5 m/s² 持续 6 帧) | 完全没有 |
| **Raw eye-tracker fixation events** (saccade / fixation segmentation) | 无 — 只有 processed gazemap PNG | 操作化 attention shift 时需要 raw fixation timing | 完全没有 |

**结论硬约束**: 即使忽略 ego CAN 缺失问题(假设我们去标速度),0.6s sparse gaze 已经把 lag measurement resolution 限制到 ±300ms 以上,**根本测不到 reaction time 量级(200–400ms)的事件**。Pilot 对照实验 reactive vs proactive 直接做不出来。

---

## 3. 三个选项

### 选 A: 下载 DR(eye)VE 原始数据集(推荐)

**来源**: http://imagelab.ing.unimore.it/dreyeve (公开,论文 [54] 仓库)

**包含**:
- 74 video × 5 min × 25 fps RGB(garmin VIRB)+ etg(eye-tracking glasses)双视角
- per-frame gaze coordinate(Tobii Pro Glasses 2,~30 Hz, mapped to garmin frame)
- ego CAN: speed / acceleration / GPS / 部分车有 steering(论文未明说有 brake 信号,需要从 deceleration 推)
- 共 ~80 GB,需要邮件申请获取下载链接(或学术联系)

**代价**: 1–3 天 wall-clock(申请 + 下载 + 解压 + 写 loader),全 CPU
**收益**: 原始 25fps gaze + ego speed/accel 全有,pilot 设计 100% 可执行

**注意**: 论文 [54] 没明确说 raw brake pedal signal 是否在公开数据里 — 如果没 raw brake,只能用 longitudinal acceleration < threshold 推 brake onset,精度足够 pilot,但失去"brake pedal force"作为 brake intensity 协变量(隐患 #2 ANCOVA 退化为只用 |a|)

### 选 B: 换数据集 — 找原生包含 gaze + CAN 的连续视频集

候选(我未验证可获取性):
- **DGaze** (Cheng et al., CVPR 2024 IVGaze 的前身?): 仓库声明 in-vehicle gaze + 车信号,但论文重点是 in-vehicle camera 不是 ego-view scene
- **LBW** (Kasahara et al., ECCV 2022): 28 drivers,自然驾驶,gaze + scene,**但 W³DA 已经把 LBW 抽稀**,跟 DR(eye)VE 同样问题。原始 LBW 是否有 CAN 需要查论文
- **HDD** (Honda Driving Dataset, Ramanishka et al., CVPR 2018): 137h 视频 + 4-tier action label + CAN,**没有 gaze**

**判断**: 没有比 DR(eye)VE 更对路的现成数据集。LBW 是潜在 fallback,但跟 DR(eye)VE 同来源问题。

### 选 C: 修改 pilot scope 用现有 W³DA sparse 做"弱版" pilot

**改造**: 不测 lag 时间,改测 **"key-frame index 关系"** —
- 在 W³DA sparse frame 序列里,第 i 个 key frame 包含 brake-relevant What/Why 文本(用 LLM 检测 keyword "brake" / "stop" / "decelerate")
- 看这种 cognitive cue 出现的 key frame index,是否早于该视频段的 ego speed drop(从 prompt 字符串提取的 speed 数字序列)
- 用 key frame 间距(15 帧 = 0.6s)做粗粒度 lag

**代价**: 1-2 天纯 CPU,完全不需要新下载
**收益**: 一个粗粒度 sanity check,看趋势是否站得住
**问题**:
- 速度从 prompt 字符串解析 = 极脆弱,只有当时刻速度,没有连续 trace
- 0.6s 时间分辨率,**测不到 reaction time 量级**,也分不开 reactive / proactive 的 200ms vs 800ms 差异
- W³DA 的 key-sample selection 本身就基于 attention shift(论文 §3.1),"key frame 相邻语义变化"是定义上保证存在的,**抽稀本身可能已经污染 lag 信号**
- reviewer 一眼看出方法有问题

**结论**: C 是 "我们想动手就能动手" 的版本,但科学上不站得住。最多作为热身让代码 pipeline 跑通,不能作为 pilot 主结果。

---

## 4. 我的建议

**选 A,联系 DR(eye)VE 作者获取原始数据**。理由:
- 这是唯一能让 pilot 严谨执行的路径
- 1-3 天等待期不阻塞 — 期间可以并行做 (1) B1 改动 4 + smoke 收尾 (2) pilot code skeleton(brake detector / fixation shift detector / mixed-effect 统计模板),数据来了直接跑
- DR(eye)VE 是 ICCV 2025 论文 [54] 的官方数据集,Cucchiara 组维护,学术请求大概率给

**Plan-B 兜底**: 如果一周内 DR(eye)VE raw 拿不到,降级到选 C 跑 sanity 但**不当 pilot 主结果**,同时联系 LBW 作者要原始 CAN。

---

## 5. 等用户决定的点

1. 选 A / B / C / 其他
2. 如果 A: 你有 DR(eye)VE 学术联系吗,还是要我起草一封请求邮件给你?
3. 等待期间,B1 dispatch 收尾 + pilot code skeleton 是否并行?
