# DSAddpm (LDCT→NDCT DDPM) — DSA/CPRAformer-Inspired Integration (Project Closed)

This folder is a research fork based on a DDPM / Fast-DDPM style codebase, adapted for **LDCT→NDCT CT denoising** and extended with a time-conditioned dual-stream block (**DSA_Block**) inspired by CPRAformer-style ideas.

**Status**: This line of work is **ended / archived**.

**Outcome**: After ~600k training steps, the modification showed **no clear improvement and no clear degradation** versus the baseline DDPM under the tested setup.

For a complete snapshot of the final configuration, reproduction commands, and result summary, see `REPORT.md`.

---

## English

### Motivation
LDCT denoising often involves (1) global structured artifacts (e.g., stripe-like patterns) and (2) delicate local anatomy (vessels/bone edges). The goal of this modification was to introduce a lightweight, time-conditioned mechanism that can address both aspects without using heavy native attention.

### Core idea (DSA_Block)
`DSA_Block` is a dual-stream block conditioned on diffusion time embedding (`time_emb`):
- **Global stream**: sparse prompt channel selection (`SPC_SA`) to focus on a Top-k subset of channels.
- **Local stream**: efficient spatial refinement (`SPR_SA`) using depthwise 3×3 + pointwise 1×1 convolutions with SiLU.
- **Fusion**: frequency-domain fusion (`AAFM`) using FFT and learnable complex weights.
- **FFN replacement**: multi-branch gated network (`MSGN`).

All sub-modules accept `time_emb` and inject it via scale/shift conditioning.

### Integration into U-Net
The original `AttnBlock` (Q/K/V full attention) can be replaced by `DSA_Block` at:
- bottleneck (mid block), and
- low-resolution attention stages (e.g., 16×16),
controlled by `configs/ldfd_linear.yml -> model.dsa`.

LDCT is concatenated as condition at the input (e.g., `[LD, x_t]`), so `model.in_channels` remains `cond_channels + noisy_channels` (default `2`).

### Reproducibility
See `REPORT.md` for the final config and commands.

### Project status
This project direction is archived due to the observed neutral result after long training.

---

## 中文

### 改进动机
LDCT 去噪通常同时存在两类挑战：
1) 全局结构化伪影（例如条纹类噪声/伪影）；
2) 局部解剖结构细节需要尽量保真（血管、骨骼边缘等）。
本工作希望在不引入过重原生注意力的前提下，加入轻量、可由扩散时间步控制的结构，兼顾全局与局部。

### 核心思想（DSA_Block）
`DSA_Block` 是一个由扩散时间嵌入（`time_emb`）条件控制的双流模块：
- **全局分支**：`SPC_SA`（Prompt 引导 Top-k 通道稀疏选择）。
- **局部分支**：`SPR_SA`（Depthwise 3×3 + Pointwise 1×1 + SiLU 的空间细化）。
- **融合**：`AAFM`（FFT 频域融合 + 可学习复数权重）。
- **前馈替换**：`MSGN`（多分支门控网络）。

所有子模块都接收 `time_emb`，并通过 scale/shift 的方式注入条件。

### 在 U-Net 中的替换位置
原始的 `AttnBlock`（Q/K/V 全注意力）可在以下位置替换为 `DSA_Block`：
- bottleneck（mid block），以及
- 低分辨率注意力层（例如 16×16），
由 `configs/ldfd_linear.yml -> model.dsa` 控制。

LDCT 作为条件与噪声图拼接输入（例如 `[LD, x_t]`），因此 `model.in_channels` 保持为 `cond_channels + noisy_channels`（默认 `2`）。

### 复现
最终配置、复现命令与结果摘要见 `REPORT.md`。

### 项目状态
该方向在长训练后结果呈中性（无明显提升/下降），因此已归档关闭。

---

## Requirements / 依赖

### Install
```bash
pip install -r requirements.txt
```

This repo includes:
- `requirements.txt`: minimal dependency list.
- `requirements-lock.txt`: a pinned snapshot of a working environment.

本仓库包含：
- `requirements.txt`：最小依赖列表。
- `requirements-lock.txt`：可运行环境的版本锁定快照。

---

## Git hygiene / Git 规范

This repo includes a `.gitignore` to avoid committing datasets, training outputs, checkpoints, and generated images.

本仓库包含 `.gitignore`，用于避免提交数据集、训练输出、权重文件与生成图片等大文件。
