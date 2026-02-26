import math
import torch
import torch.nn as nn


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(
            x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.temb_proj = torch.nn.Linear(temb_channels,
                                         out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h*w)  # b,c,hw
        w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x+h_


class TimeScaleShift(nn.Module):
    def __init__(self, channels, temb_channels):
        super().__init__()
        self.temb_proj = nn.Linear(temb_channels, channels * 2)

    def forward(self, x, temb):
        scale_shift = self.temb_proj(nonlinearity(temb))
        scale, shift = torch.chunk(scale_shift, 2, dim=1)
        return x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]


class SPC_SA(nn.Module):
    def __init__(self, channels, temb_channels, topk_ratio=0.5, prompt_hidden_ratio=2.0):
        super().__init__()
        self.channels = channels
        self.topk_ratio = topk_ratio
        hidden = max(4, int(channels * prompt_hidden_ratio))
        self.time_affine = TimeScaleShift(channels, temb_channels)
        self.time_to_prompt = nn.Linear(temb_channels, channels)
        self.prompt_mlp = nn.Sequential(
            nn.Linear(channels * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, channels),
        )
        self.q = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb):
        h = self.time_affine(x, temb)
        b, c, hgt, wgt = h.shape
        gap = h.mean(dim=(2, 3))
        time_prompt = self.time_to_prompt(nonlinearity(temb))
        prompt = self.prompt_mlp(torch.cat([gap, time_prompt], dim=1))
        k = max(1, int(c * float(self.topk_ratio)))
        _, topk_idx = torch.topk(prompt, k, dim=1)
        idx = topk_idx[:, :, None, None].expand(-1, -1, hgt, wgt)
        q = torch.gather(self.q(h), 1, idx)
        k_feat = torch.gather(self.k(h), 1, idx)
        v = torch.gather(self.v(h), 1, idx)
        q = q.reshape(b, k, hgt * wgt)
        k_feat = k_feat.reshape(b, k, hgt * wgt)
        attn = torch.bmm(q, k_feat.transpose(1, 2)) * (k ** -0.5)
        attn = torch.softmax(attn, dim=2)
        v = v.reshape(b, k, hgt * wgt)
        out = torch.bmm(attn, v).reshape(b, k, hgt, wgt)
        out_full = torch.zeros_like(h)
        out_full.scatter_(1, idx, out)
        out = self.proj_out(out_full)
        return out


class SPR_SA(nn.Module):
    def __init__(self, channels, temb_channels):
        super().__init__()
        self.time_affine = TimeScaleShift(channels, temb_channels)
        self.dw = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.act = nn.SiLU()

    def forward(self, x, temb):
        h = self.time_affine(x, temb)
        h = self.dw(h)
        h = self.act(h)
        h = self.pw(h)
        h = self.act(h)
        return h


class AAFM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.w1_real = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.w1_imag = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.w2_real = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.w2_imag = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, feat_global, feat_local):
        f1 = torch.fft.fft2(feat_global, norm="ortho")
        f2 = torch.fft.fft2(feat_local, norm="ortho")
        w1 = torch.complex(self.w1_real, self.w1_imag)
        w2 = torch.complex(self.w2_real, self.w2_imag)
        fused = f1 * w1 + f2 * w2
        out = torch.fft.ifft2(fused, norm="ortho").real
        return out


class MSGN(nn.Module):
    def __init__(self, channels, temb_channels):
        super().__init__()
        self.time_affine = TimeScaleShift(channels, temb_channels)
        self.branch_a = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.branch_b = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb):
        h = self.time_affine(x, temb)
        a = self.branch_a(h)
        b = self.branch_b(h)
        return a * torch.sigmoid(b)


class DSA_Block(nn.Module):
    def __init__(self, channels, temb_channels, topk_ratio=0.5, prompt_hidden_ratio=2.0):
        super().__init__()
        self.norm_global = Normalize(channels)
        self.norm_local = Normalize(channels)
        self.norm_ffn = Normalize(channels)
        self.spc_sa = SPC_SA(channels, temb_channels, topk_ratio=topk_ratio, prompt_hidden_ratio=prompt_hidden_ratio)
        self.spr_sa = SPR_SA(channels, temb_channels)
        self.aafm = AAFM(channels)
        self.msgn = MSGN(channels, temb_channels)

    def forward(self, x, temb):
        feat_global = self.spc_sa(self.norm_global(x), temb)
        feat_local = self.spr_sa(self.norm_local(x), temb)
        feat_attn = self.aafm(feat_global, feat_local)
        x = x + feat_attn
        x = x + self.msgn(self.norm_ffn(x), temb)
        return x


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(config.model.ch_mult)
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        in_channels = config.model.in_channels
        resolution = config.data.image_size
        resamp_with_conv = config.model.resamp_with_conv
        num_timesteps = config.diffusion.num_diffusion_timesteps
        
        if config.model.type == 'bayesian':
            self.logvar = nn.Parameter(torch.zeros(num_timesteps))
        
        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        dsa_cfg = getattr(config.model, "dsa", None)
        self.dsa_enabled = bool(getattr(dsa_cfg, "enabled", False)) if dsa_cfg is not None else bool(getattr(config.model, "dsa_enabled", False))
        self.dsa_resolutions = list(getattr(dsa_cfg, "resolutions", [])) if dsa_cfg is not None else list(getattr(config.model, "dsa_resolutions", []))
        self.dsa_use_mid = bool(getattr(dsa_cfg, "use_mid", False)) if dsa_cfg is not None else bool(getattr(config.model, "dsa_use_mid", False))
        self.dsa_topk_ratio = float(getattr(dsa_cfg, "spc_topk_ratio", 0.5)) if dsa_cfg is not None else float(getattr(config.model, "dsa_topk_ratio", 0.5))
        self.dsa_prompt_hidden_ratio = float(getattr(dsa_cfg, "spc_prompt_hidden_ratio", 2.0)) if dsa_cfg is not None else float(getattr(config.model, "dsa_prompt_hidden_ratio", 2.0))

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch,
                            self.temb_ch),
            torch.nn.Linear(self.temb_ch,
                            self.temb_ch),
        ])

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+ch_mult
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    if self.dsa_enabled and curr_res in self.dsa_resolutions:
                        attn.append(DSA_Block(block_in, self.temb_ch, topk_ratio=self.dsa_topk_ratio, prompt_hidden_ratio=self.dsa_prompt_hidden_ratio))
                    else:
                        attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        if self.dsa_enabled and self.dsa_use_mid:
            self.mid.attn_1 = DSA_Block(block_in, self.temb_ch, topk_ratio=self.dsa_topk_ratio, prompt_hidden_ratio=self.dsa_prompt_hidden_ratio)
        else:
            self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    if self.dsa_enabled and curr_res in self.dsa_resolutions:
                        attn.append(DSA_Block(block_in, self.temb_ch, topk_ratio=self.dsa_topk_ratio, prompt_hidden_ratio=self.dsa_prompt_hidden_ratio))
                    else:
                        attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x, t):
        assert x.shape[2] == x.shape[3] == self.resolution

        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    attn_module = self.down[i_level].attn[i_block]
                    if isinstance(attn_module, DSA_Block):
                        h = attn_module(h, temb)
                    else:
                        h = attn_module(h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]

        h = self.mid.block_1(h, temb)
        if isinstance(self.mid.attn_1, DSA_Block):
            h = self.mid.attn_1(h, temb)
        else:
            h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    attn_module = self.up[i_level].attn[i_block]
                    if isinstance(attn_module, DSA_Block):
                        h = attn_module(h, temb)
                    else:
                        h = attn_module(h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h
