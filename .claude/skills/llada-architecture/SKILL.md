---
name: llada-architecture
description: LLada 代码架构速查 —— forward 数据流、`[ATTN]` token 机制、各文件职责、LoRA 目标层。在任何修改 `model/Attn_model.py` 前**必须先读**,仅用于理解代码(不含具体修改步骤,那是 modify-llada)。用户问"模型怎么工作"、"热力图在哪生成"、"loss 在哪算"、"文件结构"时用。
---

# LLada 架构速查

## 什么时候用这个 skill

- 用户问"这个模型 / 这段代码是什么意思"
- 要在 `model/Attn_model.py` 里加/改代码
- 要追踪一个数据张量从输入到 loss 的完整流动
- 要判断一个 bug 是 LLaVA 侧还是 attention decoder 侧的问题

## 一、顶层数据流 (记住这个流程图)

```
image ──┬──> CLIP Vision Tower ──> visual tokens (已冻结)
        └──> preprocess(1024×1024) ──> (给 attn_decoder 用的高分辨率特征)

text ────> LLaVA Tokenizer ──> text tokens (含 [ATTN] 特殊 token)

[visual tokens, text tokens] ──> LLaVA LLM (Vicuna-7B, LoRA 微调)
       │
       ├──> lm_head  ─────────────────────> text 输出 (What + Why)
       │
       └──> 取 [ATTN] token 位置的 hidden state
              │
              └──> text_hidden_fcs (投影到 out_dim=1024)
                     │
                     └──> attn_decoder (cross-attention) ──> 热力图 (map_size=256)
```

**关键一点**: `[ATTN]` 是唯一把"语言侧的认知推理"和"像素侧的热力图"绑起来的桥梁。改它前要非常清楚后果。

## 二、文件到功能的映射

### `model/Attn_model.py` (核心)
必看 symbol(用 grep 快速定位):
- `class AttnForCausalLM(LlavaLlamaForCausalLM)` — 继承 LLaVA,加 attention decoder
- `initialize_attn_modules(self, config)` — 在 LLaVA 基础上注入 `text_hidden_fcs` + `attn_decoder`;训练时调,eval 时不调
- `def forward(...)` — 同时计算 CE loss 和 attn loss
- `def evaluate(...)` — 推理入口,chat.py 调用
- `def CC(pred, gt)`, `KLDivergence`, `SIM`, `NSS`, `AUC_J`, `AUC_B` — 6 个热力图指标
- `attn_token_idx` — `[ATTN]` 在 vocab 中的位置,默认 32003

### `model/llava/` (基座,慎改)
- `conversation.py` — `conv_templates["llava_v1"]` 是默认模板;如果你要改 prompt 格式,在这里(而不是在 dataset.py 里拼字符串)
- `mm_utils.py::tokenizer_image_token` — 把 `<image>` 占位符替换为 `IMAGE_TOKEN_INDEX=-200`
- 其他文件基本是 LLaVA 原生,**不要**改

### `utils/dataset.py`
- `class HybridDataset` — 根据 `sample_rate=[8,5,2,7]` 混合 4 个子数据集
- `class BDDA / DReyeVE / LBW / DADA` — 各子数据集的 `__getitem__`
- `def collate_fn` — 把 batch 拼起来,处理 padding、conversation 构造
- 每个样本产出的字典包含: `images_clip`, `images` (高分辨率 for attn decoder), `input_ids`, `labels`, `masks_list`(热力图 GT), `conversation_list`(文本 GT),`image_paths`

### `train_ds.py` (主循环)
- `main()` → `train()` / `validate()`
- `ds_config` 硬编码(约 250 行)—— ZeRO stage 2,AdamW β=(0.9, 0.95),WarmupDecayLR 100 步 warmup
- `validate()` 里根据 `eval_text` 决定是否跑 LLM 生成(跑了会慢 10×)
- 最佳 ckpt 判定: `kld<best_kld OR cc>best_cc OR sim>best_sim OR nss>best_nss OR auc_b>best_aucb OR auc_j>best_aucj` —— 任意一个变好都会 save

## 三、forward 里到底发生了什么 (读代码前的心智模型)

伪代码(按你将要在 `Attn_model.py::forward` 看到的顺序):

```python
def forward(self, images_clip, images, input_ids, labels, masks_list, ...):
    # 1. 视觉编码 (冻结)
    visual_feats = self.vision_tower(images_clip)           # [B, N_img, 1024]

    # 2. LLM 前向 (LoRA 微调)
    outputs = super().forward(input_ids, visual_feats, labels)
    #   → outputs.loss 是 CE loss (对 What + Why 的下一 token 预测)
    #   → outputs.hidden_states[-1] 是最后一层 hidden state

    # 3. 取 [ATTN] token 位置的 hidden state
    attn_mask = (input_ids == self.attn_token_idx)
    attn_hidden = last_hidden[attn_mask]                    # [B, hidden_dim]

    # 4. 投影 + 解码成热力图
    attn_embed = self.text_hidden_fcs(attn_hidden)          # [B, out_dim=1024]
    pred_sal = self.attn_decoder(attn_embed, images)        # [B, 256, 256]

    # 5. 三部分 loss
    attn_loss = loss_fn(pred_sal, masks_list)               # BCE / KL
    ce_loss = outputs.loss                                  # LLM 的完整 CE
    ce_what_loss, ce_why_loss = split_loss_what_why(...)    # 分段 CE

    total = (attn_loss_weight * attn_loss
           + ce_loss_weight    * ce_loss
           + ce_what_loss_weight * ce_what_loss
           + ce_why_loss_weight  * ce_why_loss)
    return {"loss": total, "pred_sal": pred_sal, "gt_sal": masks_list, ...}
```

**真实代码可能有细节出入**(比如 hidden_states 取哪一层),读 `Attn_model.py` 时以实际代码为准,但顶层结构不会变。

## 四、LoRA 到底套在哪些层

`train_ds.py::find_linear_layers`:
```python
# target: 名字包含 q_proj / v_proj 的所有 Linear
# 排除: attn_decoder, vision_tower, mm_projector, text_hidden_fcs
```
也就是说 LoRA **只套在 LLM 的 self-attention 的 Q/V 投影上**。`lm_head`、`embed_tokens`、`attn_decoder`、`text_hidden_fcs` 是**全量训练**,不经过 LoRA。

加新的全量训练模块时(比如新加一个 decoder),记得:
1. 把模块名字加进 `find_linear_layers` 的排除列表
2. 在 `train_ds.py` 的参数解冻循环里加上 `if "你的模块名" in n: p.requires_grad = True`

## 五、validate 的两种模式

| 参数 | 跑了什么 | 速度 | 用途 |
| --- | --- | --- | --- |
| `--eval_only`(仅此) | 只计算热力图 6 指标 | 快 | 训练中每个 epoch 后自动跑 / 快速验证模型改得对不对 |
| `--eval_only --eval_text` | 上面 + LLM 逐 token 生成 + BLEU/METEOR/ROUGE/CIDEr/CIDErR × {full, what, why} | 慢 10× | 最终论文表格数据 |
| `--eval_text_resume <log_name>` | 复用之前生成好的文本 | 最快 | 只想重新算文本指标时 |

每个样本的文本生成结果会 save 到 `<原视频目录>/eval_text/<log_name>/<frame>.txt`(如果开了 `--eval_text_save`)。

## 六、改代码时的"血泪清单"

在动 `model/Attn_model.py` 之前问自己:

1. 我的改动会影响 `[ATTN]` token 的 hidden state 吗? 如果会,`text_hidden_fcs` 的 input dim 是否也要改?
2. 我的改动会改变 `pred_sal` 的 shape 吗? 如果会,`masks_list`(GT)的处理和 6 个指标函数是否都兼容?
3. 我加的参数在 LoRA 还是全量? 用 `model.print_trainable_parameters()` 验证
4. 我的改动会让某个 rank 走不同分支吗? 分布式训练会 hang
5. 新增的 loss 有没有加到 `total_loss` 里? 有没有在 `AverageMeter` 注册?

## 七、一分钟 onboarding

如果你刚接手这个仓库,按这个顺序读:

1. `train_ds.py::main` 前 100 行(了解参数/模型加载)
2. `train_ds.py::train` 循环(了解训练一步做了什么)
3. `model/Attn_model.py::AttnForCausalLM.__init__` 和 `initialize_attn_modules`
4. `utils/dataset.py::HybridDataset.__getitem__` 和 `collate_fn`(了解输入格式)
5. `model/Attn_model.py::forward`(核心)
6. 最后才读 `chat.py` / `merge_lora_weights_and_save_hf_model.py`
