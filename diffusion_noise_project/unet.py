
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(nn.Linear(dim, dim*4), nn.SiLU(), nn.Linear(dim*4, dim*4))
    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / (half-1))
        args = t[:,None].float() * freqs[None]
        return self.proj(torch.cat([torch.sin(args), torch.cos(args)], dim=-1))

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, num_groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(num_groups, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(num_groups, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.act = nn.SiLU()
        self.time_proj = nn.Linear(time_emb_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x, t_emb):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_proj(self.act(t_emb))[:,:,None,None]
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, model_channels=64, channel_mults=(1,2,4)):
        super().__init__()
        time_emb_dim = model_channels * 4
        self.time_emb = SinusoidalTimeEmbedding(model_channels)
        chs = [model_channels * m for m in channel_mults]
        self.input_conv = nn.Conv2d(in_channels, chs[0], 3, padding=1)
        self.enc1 = ResBlock(chs[0], chs[0], time_emb_dim)
        self.down1 = nn.Conv2d(chs[0], chs[0], 4, stride=2, padding=1)
        self.enc2 = ResBlock(chs[0], chs[1], time_emb_dim)
        self.down2 = nn.Conv2d(chs[1], chs[1], 4, stride=2, padding=1)
        self.enc3 = ResBlock(chs[1], chs[2], time_emb_dim)
        self.mid1 = ResBlock(chs[2], chs[2], time_emb_dim)
        self.mid2 = ResBlock(chs[2], chs[2], time_emb_dim)
        self.up3 = nn.ConvTranspose2d(chs[2], chs[2], 4, stride=2, padding=1)
        self.dec3 = ResBlock(chs[2]+chs[1], chs[1], time_emb_dim)
        self.up2 = nn.ConvTranspose2d(chs[1], chs[1], 4, stride=2, padding=1)
        self.dec2 = ResBlock(chs[1]+chs[0], chs[0], time_emb_dim)
        self.dec1 = ResBlock(chs[0]+chs[0], chs[0], time_emb_dim)
        self.out_norm = nn.GroupNorm(8, chs[0])
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(chs[0], in_channels, 1)
    def forward(self, x, t):
        t_emb = self.time_emb(t)
        h0 = self.input_conv(x)
        h1 = self.enc1(h0, t_emb); h1d = self.down1(h1)
        h2 = self.enc2(h1d, t_emb); h2d = self.down2(h2)
        h3 = self.enc3(h2d, t_emb)
        h = self.mid2(self.mid1(h3, t_emb), t_emb)
        h = F.interpolate(self.up3(h), size=h2.shape[2:], mode="nearest")
        h = self.dec3(torch.cat([h, h2], dim=1), t_emb)
        h = F.interpolate(self.up2(h), size=h1.shape[2:], mode="nearest")
        h = self.dec2(torch.cat([h, h1], dim=1), t_emb)
        h = self.dec1(torch.cat([h, h0], dim=1), t_emb)
        return self.out_conv(self.out_act(self.out_norm(h)))
