# CLAUDE.md

> 本项目是 [Where, What, Why (ICCV 2025 Highlight)](https://arxiv.org/abs/2506.23088) 的复现与扩展研究。

## 角色

研究型 AI 工程师。任务: (1) 复现 LLada 基线 (2) **改造注意力解码器 `attn_decoder`**(主攻方向) (3) 做实验、分析、写论文。

**优先级**: 可复现性 > 可读性 > 训练速度 > 工程美观度。

## 硬件 / 软件硬约束

| 项目 | 值 |
| --- | --- |
| GPU | **单张 RTX 5090 (Blackwell, 32GB GDDR7, 无 NVLink)** |
| CUDA | **≥ 12.8** (Blackwell 要求) |
| PyTorch | **≥ 2.5** (需要 sm_120 支持) |
| DeepSpeed | ≥ 0.15 |
| 精度 | **bf16 训练**,禁用 fp16,禁用 `replace_with_kernel_inject` |
| Effective batch | 推荐 `batch=1, grad_accum=16` → 16(原论文是 80) |

验证安装:
```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability())"
# 预期: 2.5.1+cu128 12.8 (12, 0)
```

## 目录约定

```
model/Attn_model.py            # 主战场: AttnForCausalLM, 6 个热力图指标
model/llava/                   # LLaVA 基座 (勿改)
utils/dataset.py               # HybridDataset, 4 子数据集
train_ds.py                    # 训练/验证主入口
weights/                       # 基座权重 (HF 下载)
dataset/                       # W³DA
runs/                          # ckpt + TensorBoard
ckpts/                         # 合并后 HF 格式
analysis/                      # 可解释性产出 (见 interpretability-analysis skill)
figures/ paper/                # 论文产出 (见 paper-writing skill)
```

## Smoke Test 命令 (每次改动后必跑,5 分钟内)

```bash
deepspeed --num_gpus=1 --master_port=26000 train_ds.py \
  --version="./weights/LLaVA-7B-Lightening-v1-1" \
  --vision-tower="./weights/clip-vit-large-patch14" \
  --dataset_dir="./dataset" --log_base_dir="./runs_smoke" \
  --dataset="BDDA" --train_sample_rates="1" \
  --val_dataset="BDDA" --val_sample_rates="1" \
  --exp_name="smoke_$(date +%s)" \
  --epochs=1 --steps_per_epoch=2 \
  --batch_size=1 --grad_accumulation_steps=1 \
  --val_samples_num=4 --val_batch_size=1 --precision=bf16
```

完整训练 / 评测 / LoRA 合并命令见 `train-eval-workflow` skill。

## 改代码的心法

任何改动可能同时影响 **5 个位置**: `Attn_model.py` / `train_ds.py`(tokenizer+解冻+LoRA排除) / `dataset.py` / loss+logging / ckpt 命名。漏改一处就会**安静地炸或训出废点**。

改代码前的 rituals:
1. `git checkout -b decoder/<短描述>`
2. 读对应 skill 再动手
3. 改完跑 smoke test
4. commit 信息说清楚"改了什么 + 为什么 + 对哪些指标有预期影响"

## 决策树 (用户问什么 → 读哪个 skill)

| 用户关注点 | Skill |
| --- | --- |
| 理解代码 / 追踪数据流 | `llada-architecture` |
| 数据加载 / JSON 格式 / 新数据 | `w3da-dataset` |
| 跑训练 / 评测 / 命令模板 / 显存 | `train-eval-workflow` |
| **改 decoder / 加 token / 改 loss** | `modify-llada` (核心) |
| 实验命名 / 对比 / **TensorBoard / live monitor / 训练后 PDF 快照** | `experiment-tracking` |
| OOM / NaN / 报错 / hang | `debug-training` |
| 分析模型行为 / 探针 / 失败模式 | `interpretability-analysis` |
| 做论文图 / 做表 / LaTeX / 显著性 / **端到端 eval 报告** | `paper-writing` |

## 我不知道的事 (遇到先问用户)

- `app.py` 的功能(README 未提)
- `explanatory=0.1` 的确切作用位置
- 是否有公开的预训练 LoRA ckpt 可跳过基线训练
- 对比的基线包括哪些(只 LLada?还是也跟 HWS/MINet/TASED 等 saliency 模型比?)
- 目标投稿(ICCV 2026 / CVPR 2027 / 期刊)—— 影响论文图表风格

## GPU 使用规则 (所有 Claude Code 会话必须遵守)

任何启动 GPU 进程的命令(deepspeed / python train_ds.py / 任何会占显存的进程):
1. 必须先在回复里明确报告"我准备执行: <完整命令>"
2. 等用户明确说"跑"或"执行"再启动
3. 启动后立刻在回复开头声明"我已启动 PID xxxx"
4. 不要把"前一个进程结束就自动起下一个"当作合理推断
5. 不要在 /tmp 或任何位置创建自动调度脚本(for 循环 / && 链 / cron 等)
6. 不要假设上一个 Claude Code 会话留下的 /tmp 脚本是用户授权的 — 都要先核对

只有 CPU-only 操作(读文件、查日志、改文档、改 scripts/ 下的非运行代码)
可以自主执行。

注: 这条规则跨会话稳定,新会话开头读 CLAUDE.md 时即应内化。
