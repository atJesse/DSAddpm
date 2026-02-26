# DSAddpm — DSA Block Integration (LDCT→NDCT DDPM) Final Report

Date: 2026-02-26

---

## 1) English

### Scope
This work integrates a lightweight, time-conditioned dual-stream attention block (**DSA_Block**) into an existing LDCT→NDCT DDPM U-Net. The implementation targets stripe/artifact suppression and local detail preservation while keeping memory usage reasonable (no heavy full-resolution native attention).

### What Changed
**Files**
- `models/diffusion.py`
  - Added modules: `SPC_SA`, `SPR_SA`, `AAFM`, `MSGN`, `DSA_Block`.
  - All new modules accept `time_emb` (time-step embedding) and inject it via scale/shift conditioning.
  - Replaced selected `AttnBlock` instances with `DSA_Block` at configured resolutions and/or bottleneck.
- `configs/ldfd_linear.yml`
  - Added `model.dsa` section to control enabling, placement, and hyperparameters.

**Original vs New (Conceptually)**
- Before: U-Net used `AttnBlock` (Q/K/V full attention) in the attention resolutions and bottleneck; `AttnBlock` only took `x`.
- After: When enabled by config, those attention slots can be replaced by `DSA_Block` which is time-conditioned and composed of:
  - `SPC_SA`: sparse prompt channel selection (Top-k channels; k controlled by `spc_topk_ratio`).
  - `SPR_SA`: efficient spatial refinement (depthwise 3×3 + pointwise 1×1, with SiLU).
  - `AAFM`: frequency-domain fusion (FFT → learnable complex weights → IFFT).
  - `MSGN`: multi-branch gating network as FFN replacement (3×3 branch + 1×1 branch + gating).

### Final Experiment Configuration (LDCT)
The final configuration was recorded in `configs/ldfd_linear.yml`. Key parts:

```yaml
data:
  dataset: "LDFDCT"
  train_dataroot: "data/LD_FD_CT_train"
  sample_dataroot: "data/LD_FD_CT_test"
  image_size: 256
  channels: 1

model:
  type: "sg"
  in_channels: 2            # [LD, x_t] concatenation
  out_ch: 1
  ch: 128
  ch_mult: [1, 1, 2, 2, 4, 4]
  num_res_blocks: 2
  attn_resolutions: [16]
  ema: True
  dsa:
    enabled: True
    resolutions: [16]
    use_mid: True
    spc_topk_ratio: 0.5
    spc_prompt_hidden_ratio: 2.0

training:
  batch_size: 8
  snapshot_freq: 10000

optim:
  optimizer: "Adam"
  lr: 0.00002
```

### Reproduction Commands
**Environment**
```bash
conda activate D2Dfastddpm
cd DSAddpm
```

**Train (DDPM, 1000 steps)**
```bash
nohup python3 ddpm_main.py \
  --config ldfd_linear.yml \
  --dataset LDFDCT \
  --exp ./train_out \
  --doc ldfd_ddpm_run1 \
  --timesteps 1000 \
  > ./train_out/train.log 2>&1 &
```

**Resume training**
```bash
nohup python3 ddpm_main.py \
  --config ldfd_linear.yml \
  --dataset LDFDCT \
  --exp ./train_out \
  --doc ldfd_ddpm_run1 \
  --timesteps 1000 \
  --resume_training \
  > ./train_out/train.log 2>&1 &
```

**Sampling / evaluation**
This codebase supports sampling via the `--sample --fid` flags. Use the same config and experiment directory:
```bash
nohup python3 ddpm_main.py \
  --config ldfd_linear.yml \
  --dataset LDFDCT \
  --exp ./train_out \
  --doc ldfd_ddpm_run1 \
  --sample --fid \
  --timesteps 1000 \
  > ./train_out/sample.log 2>&1 &
```

### Observed Result Summary
- After training for **~600k steps**, no clear improvement was observed compared with the baseline DDPM (no obvious gain, and no obvious degradation).
- The modification therefore did not justify further iteration under the current training setup and hyperparameters.

### Recommended Ablations (Config-Only)
These can be run without changing code:
- Baseline: `model.dsa.enabled: False`
- Only bottleneck DSA: `enabled: True`, `use_mid: True`, `resolutions: []`
- Only low-resolution DSA (e.g., 16×16): `enabled: True`, `use_mid: False`, `resolutions: [16]`
- Full DSA: `enabled: True`, `use_mid: True`, `resolutions: [16]`
- Top-k sweep: `spc_topk_ratio ∈ {0.25, 0.5, 0.75}`

---

## 2) 中文

### 工作范围
本工作在现有 LDCT→NDCT 的 DDPM U-Net 框架中集成轻量、可接收时间步条件的双流注意力块（**DSA_Block**）。目标是：在不引入过重原生全注意力的前提下，加强条纹/伪影的全局建模与局部细节保真。

### 改动内容
**改动文件**
- `models/diffusion.py`
  - 新增模块：`SPC_SA`、`SPR_SA`、`AAFM`、`MSGN`、`DSA_Block`。
  - 所有新模块均接收 `time_emb`（时间步嵌入），并通过 scale/shift 的方式注入条件。
  - 在配置指定的分辨率层与/或 bottleneck 位置，将原 `AttnBlock` 替换为 `DSA_Block`。
- `configs/ldfd_linear.yml`
  - 新增 `model.dsa` 配置段，控制启用开关、替换位置与超参数。

**原始结构 vs 新结构（概念层面）**
- 改动前：U-Net 在 `attn_resolutions` 与 bottleneck 使用 `AttnBlock`（Q/K/V 全注意力），`AttnBlock` 仅输入 `x`。
- 改动后：当配置启用时，这些 attention 插槽可替换为 `DSA_Block`，并由以下组件组成：
  - `SPC_SA`：Prompt 引导的 Top‑k 通道稀疏选择（`spc_topk_ratio` 控制保留比例）。
  - `SPR_SA`：高效空间细化（Depthwise 3×3 + Pointwise 1×1，SiLU 激活）。
  - `AAFM`：频域融合（FFT → 可学习复数权重 → IFFT）。
  - `MSGN`：替代传统 FFN 的门控网络（3×3 分支 + 1×1 分支 + gating）。

### 最终实验配置（LDCT）
最终配置保存在 `configs/ldfd_linear.yml`，关键字段如下：

```yaml
data:
  dataset: "LDFDCT"
  train_dataroot: "data/LD_FD_CT_train"
  sample_dataroot: "data/LD_FD_CT_test"
  image_size: 256
  channels: 1

model:
  type: "sg"
  in_channels: 2            # 输入为 [LD, x_t] 拼接
  out_ch: 1
  ch: 128
  ch_mult: [1, 1, 2, 2, 4, 4]
  num_res_blocks: 2
  attn_resolutions: [16]
  ema: True
  dsa:
    enabled: True
    resolutions: [16]
    use_mid: True
    spc_topk_ratio: 0.5
    spc_prompt_hidden_ratio: 2.0

training:
  batch_size: 8
  snapshot_freq: 10000

optim:
  optimizer: "Adam"
  lr: 0.00002
```

### 复现命令
**环境**
```bash
conda activate D2Dfastddpm
cd DSAddpm
```

**训练（DDPM, 1000 步）**
```bash
nohup python3 ddpm_main.py \
  --config ldfd_linear.yml \
  --dataset LDFDCT \
  --exp ./train_out \
  --doc ldfd_ddpm_run1 \
  --timesteps 1000 \
  > ./train_out/train.log 2>&1 &
```

**断点继续训练**
```bash
nohup python3 ddpm_main.py \
  --config ldfd_linear.yml \
  --dataset LDFDCT \
  --exp ./train_out \
  --doc ldfd_ddpm_run1 \
  --timesteps 1000 \
  --resume_training \
  > ./train_out/train.log 2>&1 &
```

**推理采样 / 评估**
该代码支持通过 `--sample --fid` 进行采样与指标统计：
```bash
nohup python3 ddpm_main.py \
  --config ldfd_linear.yml \
  --dataset LDFDCT \
  --exp ./train_out \
  --doc ldfd_ddpm_run1 \
  --sample --fid \
  --timesteps 1000 \
  > ./train_out/sample.log 2>&1 &
```

### 结果摘要
- 训练至 **约 60 万步** 后，与 baseline DDPM 相比，未观察到明确提升（提升不明显，下降也不明显）。
- 在当前训练设置与超参下，该改动未体现出继续优化的收益。

### 建议消融（仅改配置即可）
- Baseline：`model.dsa.enabled: False`
- 仅 bottleneck：`enabled: True`, `use_mid: True`, `resolutions: []`
- 仅低分辨率层（如 16×16）：`enabled: True`, `use_mid: False`, `resolutions: [16]`
- 完整 DSA：`enabled: True`, `use_mid: True`, `resolutions: [16]`
- Top‑k 扫描：`spc_topk_ratio ∈ {0.25, 0.5, 0.75}`
