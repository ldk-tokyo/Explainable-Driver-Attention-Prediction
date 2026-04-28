# Pattern A: 加新 Special Token

典型用例: `[RISK]` 输出风险分、`[INTENT]` 输出意图分类、`[DANGER]` 输出紧急程度等。

## 核心原理

和 `[ATTN]` 同构: 在 prompt 里埋一个 special token,取该 token 的 hidden state 过一个 head 得到新的预测。

## 完整改动清单 (9 处)

### 1. `train_ds.py::main`
```python
num_added = tokenizer.add_tokens("[ATTN]")  # 已有
num_added += tokenizer.add_tokens("[RISK]") # 新增
args.attn_token_idx = tokenizer("[ATTN]", add_special_tokens=False).input_ids[0]
args.risk_token_idx = tokenizer("[RISK]", add_special_tokens=False).input_ids[0]

# ...
model_args["risk_token_idx"] = args.risk_token_idx
# ...
# 已有 resize, 无需改
```

### 2. `model/Attn_model.py::AttnForCausalLM.__init__`
```python
self.attn_token_idx = kwargs.pop("attn_token_idx")
self.risk_token_idx = kwargs.pop("risk_token_idx", None)  # ← 新增 (optional)
```

### 3. `initialize_attn_modules` 加 head
```python
self.risk_head = nn.Sequential(
    nn.Linear(config.out_dim, config.out_dim // 2),
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(config.out_dim // 2, 1),  # 回归 (risk score 0-1)
    # 分类: nn.Linear(..., num_classes) + CrossEntropy
)
```

### 4. `forward` 取 hidden state 算 loss
```python
# 在 forward 里,原有 attn 部分之后加:
if self.risk_token_idx is not None and "risk_gt" in kwargs:
    risk_mask = (input_ids == self.risk_token_idx)   # [B, L]
    risk_hidden = last_hidden_states[risk_mask]       # [sum(mask), hidden]

    # 取每个样本第一个 [RISK] 的 hidden
    # (简化: 假设每样本只有一个 [RISK])
    risk_pred = self.risk_head(risk_hidden).squeeze(-1)  # [B]

    risk_loss = F.mse_loss(risk_pred, kwargs["risk_gt"].float())

    output_dict["risk_loss"] = risk_loss
    output_dict["risk_pred"] = risk_pred

    total_loss = total_loss + self.risk_loss_weight * risk_loss
```

### 5. `utils/dataset.py` 加 GT 字段
```python
# __getitem__ 里
json_path = self.json_paths[idx]
with open(json_path) as f:
    meta = json.load(f)
# 假设 JSON 里已有 "risk" 字段 (0-1 float); 如果没有要先离线标注或规则生成
risk_gt = meta.get("risk", 0.0)

return {
    # ... 原有字段
    "risk_gt": torch.tensor(risk_gt, dtype=torch.float32),
}
```

### 6. `collate_fn` 堆叠
```python
def collate_fn(batch, ...):
    # ... 原有 stacking
    risk_gts = torch.stack([b["risk_gt"] for b in batch])
    return {
        # ...
        "risk_gt": risk_gts,
    }
```

### 7. Prompt 模板加 `[RISK]`
`model/llava/conversation.py` 里改 `llava_v1` 的 template,或在 `dataset.py` 构造 conversation 时:
```python
# 原: "1. Where: [ATTN]\n2. What: ...\n3. Reason: ..."
# 新: "1. Where: [ATTN]\n2. What: ...\n3. Reason: ...\n4. Risk: [RISK]"
```

### 8. 更新 `sep_what_and_why`
如果新 prompt 加了 "4. Risk",原 `sep_what_and_why` 的分界逻辑要重审。可能需要新写 `sep_what_why_risk`。

### 9. 参数可训练 + LoRA 排除
```python
# train_ds.py 参数解冻循环 (~第 160 行)
if any(x in n for x in ["lm_head", "embed_tokens", "attn_decoder",
                         "text_hidden_fcs", "risk_head"]):  # ← risk_head
    p.requires_grad = True

# LoRA 排除列表 (find_linear_layers)
and all(x not in name for x in [
    "attn_decoder", "vision_tower", "mm_projector",
    "text_hidden_fcs", "risk_head"  # ← risk_head
])
```

## 额外注意

### Logging
- 加 `risk_loss_meter = AverageMeter("RiskLoss", ":.4f")`
- TensorBoard `writer.add_scalar("train/risk_loss", ...)`

### Validation
- `validate()` 里计算 MAE / R² 等 risk 相关指标
- 如果是分类,加 accuracy / F1

### 数据标注
- 如果 W³DA 原 JSON 没有 risk 标注,你要先决定怎么生成:
  - **规则**: 基于事故数据集 (DADA) 的时间戳,越接近事故发生的帧 risk 越高
  - **人工**: 雇人标一个子集
  - **弱监督**: 用别的模型(如事故检测模型)打伪标签

## Smoke test

```bash
deepspeed --num_gpus=1 ... --epochs=1 --steps_per_epoch=2 ...
```

在 smoke 里额外 print:
```python
print(f"[DEBUG] risk_pred: {output_dict['risk_pred']}")
print(f"[DEBUG] risk_loss: {output_dict['risk_loss']}")
```
确认 pred 不是全 0/NaN,loss 有梯度。
