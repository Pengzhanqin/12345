import torch
import torch.nn as nn
import torch.nn.functional as F

class SqueezeExciteBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1)
        return x * w


class GraphDilatedConv(nn.Module):
    def __init__(self, in_dim, out_dim, dilation=1):
        super().__init__()
        self.dilation = dilation
        self.attn_fc = nn.Linear(2 * in_dim, 1)
        self.linear = nn.Linear(in_dim, out_dim * 2)
        self.se = SqueezeExciteBlock(out_dim)
        self.temporal_conv = nn.Conv1d(out_dim, out_dim, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)

    def _build_dilated_adj(self, T, device):
        adj = torch.zeros(T, T, device=device)
        for i in range(T):
            for j in range(T):
                if i != j and abs(i - j) % self.dilation == 0:
                    adj[i, j] = 1
        return adj

    def forward(self, x):
        B, C_in, T = x.size()
        x = x.permute(0, 2, 1)
        adj = self._build_dilated_adj(T, x.device)
        outputs = []

        for b in range(B):
            h = x[b]
            neighbors = []
            for i in range(T):
                idx = adj[i].nonzero(as_tuple=False).squeeze()
                if idx.numel() == 0:
                    neighbors.append(h[i])
                    continue
                elif idx.dim() == 0:
                    idx = idx.unsqueeze(0)
                h_i = h[i].repeat(len(idx), 1).float()
                h_j = h[idx].float()
                concat = torch.cat([h_i, h_j], dim=1)
                e_ij = self.attn_fc(concat).squeeze()
                attn = F.softmax(e_ij, dim=0).unsqueeze(1)
                agg = torch.sum(attn * h_j, dim=0)
                neighbors.append(agg)
            neighbors = torch.stack(neighbors, dim=0)
            projected = self.linear(neighbors)
            out = F.glu(projected, dim=-1)
            outputs.append(out)

        out = torch.stack(outputs, dim=0).permute(0, 2, 1)
        out = self.temporal_conv(out)
        out = self.pool(out)[:, :, :T]
        return self.se(out)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, padding='same', groups=in_channels, dilation=dilation)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class CircularTCN(nn.Module):
    def __init__(self, in_channels, out_channels=8, kernel_size=3, dilation=1):
        super().__init__()
        self.circular_pad = (kernel_size - 1) // 2 * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=0)

    def forward(self, x):
        x = torch.cat([x[:, :, -self.circular_pad:], x, x[:, :, :self.circular_pad]], dim=2)
        return self.conv(x)


class MobileNetTCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.expand = nn.Conv1d(in_channels, in_channels * 2, kernel_size=1)
        self.depthwise = DepthwiseSeparableConv(in_channels * 2, in_channels * 2)
        self.circular = CircularTCN(in_channels * 2, in_channels * 2)
        self.project = nn.Conv1d(in_channels * 2, out_channels, kernel_size=1)
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.expand(x)
        x = self.act(x)
        x = self.depthwise(x)
        x = self.circular(x)
        x = self.project(x)
        return self.act(x + residual)


class SparseAttention(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.dropout = nn.Dropout(dropout)
        self.scale = dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        B, T, D = x.shape
        H = self.heads
        topk = max(1, int(T * 0.3))
        qkv = self.qkv(x).reshape(B, T, 3, H, D // H).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        topk_scores, indices = torch.topk(scores, topk, dim=-1)
        mask = torch.full_like(scores, float('-inf'))
        mask.scatter_(-1, indices, topk_scores)
        attn = F.softmax(mask, dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.out(out)


class ConvGroup(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.depthwise = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.pointwise = nn.Conv1d(dim, dim, kernel_size=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pointwise(self.depthwise(x))
        return (x * torch.sigmoid(x)).permute(0, 2, 1)


class SparseTransformer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.embedding = nn.Linear(input_dim, 256)
        self.pos_enc = nn.Parameter(torch.randn(1, 256, 256))
        self.attn = SparseAttention(256)
        self.conv_group = ConvGroup(256)
        self.norm = nn.LayerNorm(256)
        self.fc = nn.Linear(256, 64)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        B, T, _ = x.shape
        pos = self.pos_enc[:, :T, :]
        x = self.embedding(x) + pos
        x = self.attn(x)
        x = self.conv_group(x)
        x = self.norm(x)
        x = self.fc(x)
        return x.permute(0, 2, 1)


class FusionLayer(nn.Module):
    def __init__(self, rhythm_dim, pitch_dim, output_dim):
        super().__init__()
        self.conv1d = nn.Conv1d(rhythm_dim + pitch_dim, output_dim, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(output_dim, output_dim // 2)
        self.fc2 = nn.Linear(output_dim // 2, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.act = nn.GELU()

    def forward(self, rhythm, pitch):
        x = torch.cat([rhythm, pitch], dim=1)
        x = self.conv1d(x).permute(0, 2, 1)
        x = self.act(self.fc1(x))
        x = self.norm(self.fc2(x))
        return self.act(x).permute(0, 2, 1)
