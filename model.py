import torch.nn as nn
from model_components import GraphDilatedConv, MobileNetTCN, SparseTransformer, FusionLayer

class MusicGenModel(nn.Module):
    def __init__(self, input_dim=130, tgdc_out=64, fused_dim=128):
        super().__init__()
        self.tgdc = GraphDilatedConv(input_dim, tgdc_out, dilation=2)
        self.rhythm_gen = MobileNetTCN(tgdc_out, fused_dim)
        self.pitch_transformer = SparseTransformer(input_dim=fused_dim, output_dim=64)
        self.fusion = FusionLayer(fused_dim, 64, 128)
        self.rhythm_head = nn.Conv1d(128, 8, kernel_size=1)
        self.pitch_head = nn.Conv1d(128, 1, kernel_size=1)

    def forward(self, x):
        x = self.tgdc(x)
        x_rhythm = self.rhythm_gen(x)
        x_pitch = self.pitch_transformer(x_rhythm)
        fused = self.fusion(x_rhythm, x_pitch)
        rhythm_out = self.rhythm_head(fused)
        pitch_out = self.pitch_head(fused)
        return rhythm_out, pitch_out
