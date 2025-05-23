import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)

import pytorch_model_summary

__all__ = ['CNN_small_dropout', 'MLPNN', 'SHCNN_'] 
class BaseMLP(nn.Module):
    def __init__(self):
        super(BaseMLP, self).__init__()

    def get_activation(self, name):
        activations = {
            "SiLU": nn.SiLU(inplace=True),
            "Sigmoid": nn.Sigmoid(),
            "Tanh": nn.Tanh(),
            "ELU": nn.ELU(inplace=True),
            "LeakyReLU": nn.LeakyReLU(inplace=True),
            "Mish": nn.Mish(inplace=True),
            "SeLU": nn.SELU(inplace=True),
            "ReLU": nn.ReLU(inplace=True),
            "ReLU6": nn.ReLU6(inplace=True),
            "None": nn.Identity(inplace=True),
        }
        if name in activations:
            return activations[name]
        raise ValueError(f"Invalid activation: {name}")
    
class MLPNN(BaseMLP):
    def __init__(
        self,
        input_size,
        node_num,
        output_size,
        num_layers,
        hidden_activation,
        output_activation,
        dropout_rate,
    ):
        super(MLPNN, self).__init__()
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(input_size, node_num))
        self.layers.append(self.get_activation(hidden_activation))
        self.layers.append(nn.Dropout(dropout_rate))

        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(node_num, node_num))
            self.layers.append(self.get_activation(hidden_activation))
            self.layers.append(nn.Dropout(dropout_rate))

        self.layers.append(nn.Linear(node_num, output_size))
        self.layers.append(self.get_activation(output_activation))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
class CNN_small_dropout(nn.Module):

    def __init__(self,num_DV):
        super(CNN_small_dropout, self).__init__()
        BN_momentum = 0.2
        dropout_rate = 0.2

        self.start_ch = 1024  # 초기 채널 수 수정
        self.padding_param = 1
        self.kernel_size = 3
        self.stride = 1
        
        self.fc = nn.Sequential(
            # nn.Linear(in_features=num_DV, out_features=self.start_ch),        # (batch_size, 2048)
            # nn.BatchNorm1d(self.start_ch, momentum=BN_momentum),
            # nn.SiLU(),
            # nn.Dropout(dropout_rate),
            nn.Linear(in_features=num_DV, out_features=self.start_ch * 2 * 2),        # (batch_size, 2048)
            nn.BatchNorm1d(self.start_ch*2*2, momentum=BN_momentum),
            nn.SiLU(),
            nn.Dropout(dropout_rate)
            )

        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(self.start_ch, self.start_ch // 2, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding_param),  # (batch_size, 256, 4, 4)
            nn.BatchNorm2d(self.start_ch // 2, momentum=BN_momentum),
            nn.SiLU(),
            nn.Dropout(dropout_rate),

            nn.ConvTranspose2d(self.start_ch // 2, self.start_ch // 2, kernel_size=self.kernel_size, stride=2, padding=self.padding_param),  # (batch_size, 256, 4, 4)
            nn.BatchNorm2d(self.start_ch // 2, momentum=BN_momentum),
            nn.SiLU(),
            nn.Dropout(dropout_rate),

            nn.ConvTranspose2d(self.start_ch // 2, self.start_ch // 4, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding_param),  # (batch_size, 256, 6, 6)
            nn.BatchNorm2d(self.start_ch // 4, momentum=BN_momentum),
            nn.SiLU(),
            nn.Dropout(dropout_rate),

            nn.ConvTranspose2d(self.start_ch // 4, self.start_ch // 4, kernel_size=self.kernel_size, stride=2, padding=self.padding_param),  # (batch_size, 256, 4, 4)
            nn.BatchNorm2d(self.start_ch // 4, momentum=BN_momentum),
            nn.SiLU(),
            nn.Dropout(dropout_rate),

            nn.ConvTranspose2d(self.start_ch // 4, self.start_ch // 8, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding_param),  # (batch_size, 256, 6, 6)
            nn.BatchNorm2d(self.start_ch // 8, momentum=BN_momentum),
            nn.SiLU(),
            nn.Dropout(dropout_rate),

            nn.ConvTranspose2d(self.start_ch // 8, self.start_ch // 8, kernel_size=self.kernel_size, stride=1, padding=self.padding_param),  # (batch_size, 128, 13, 13)
            nn.BatchNorm2d(self.start_ch // 8, momentum=BN_momentum),
            nn.SiLU(),
            nn.Dropout(dropout_rate),

            nn.ConvTranspose2d(self.start_ch // 8, self.start_ch // 16, kernel_size=self.kernel_size, stride=self.stride, padding=0),  # (batch_size, 128, 13, 13)
            nn.BatchNorm2d(self.start_ch // 16, momentum=BN_momentum),
            nn.SiLU(),
            nn.Dropout(dropout_rate),

            nn.ConvTranspose2d(self.start_ch // 16, self.start_ch // 16, kernel_size=4, stride=2, padding=1),  # (batch_size, 128, 28, 28)
            nn.BatchNorm2d(self.start_ch // 16, momentum=BN_momentum),
            nn.SiLU(),
            nn.Dropout(dropout_rate)
        )

        # 출력층 수정
        self.conv_last = nn.Sequential(
            nn.ConvTranspose2d(self.start_ch // 16, 6, kernel_size=3, stride=1, padding=0),  # (batch_size, 1, 32, 32)
            nn.Sigmoid()
            # nn.ReLU()

        ) # Sigmoid

    def forward(self, input):
        # 완전 연결층 통과 후 재구성
        x = self.fc(input)  # (batch_size, 2048)
        x = x.view(-1, self.start_ch, 2, 2)  # (batch_size, 512, 2, 2)
        x = self.conv5(x)  # (batch_size, 128, 28, 28)
        # x = self.conv_last(x)
        x = self.conv_last(x).view([-1,6,16,16])
        return x
    
class SHCNN_(BaseMLP):
    """설계변수 14 (or 17) → (B, 1, 31, 31) 맵핑."""

    def __init__(
        self,
        num_DV: int = 17,
        dropout_rate: float = 0.1,
        BN_momentum: float = 0.1,
        start_ch: int = 512,
        embedding_dim1: int = 512,
        embedding_dim2: int = 512,
        activation: str = "ELU",
    ) -> None:
        super(SHCNN_,self).__init__()

        # ────────────── 입력 임베딩 ──────────────
        seg1_dim, seg2_dim, seg3_dim, seg4_dim = 7, 2, 5, 1
        act = self.get_activation(activation)
        self.start_ch = start_ch
        self.dropout_rate = dropout_rate
        self.BN_momentum = BN_momentum
        self.embed1_dim1 = embedding_dim1
        self.embed2_dim2 = embedding_dim2
        
        def _embed(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim, momentum=self.BN_momentum),
                act,
                nn.Linear(out_dim, out_dim),
                nn.BatchNorm1d(out_dim, momentum=self.BN_momentum),
                act,
                nn.Dropout(self.dropout_rate),
            )

        self.embed1 = _embed(seg1_dim, self.embed1_dim1)
        self.embed2 = _embed(seg2_dim, self.embed2_dim2)
        self.embed3 = _embed(seg3_dim, self.embed1_dim1)  # 추가 임베딩 레이어 (예시로 사용)
        self.embed4 = _embed(seg4_dim, self.embed2_dim2)  # 추가 임베딩 레이어 (예시로 사용)

        embed_total = self.embed1_dim1 + self.embed2_dim2

        # ────────────── Latent FC ──────────────
        self.fc = nn.Sequential(
            nn.Linear(embed_total, self.start_ch * 4),       # 2×2, C = self.start_ch
            nn.BatchNorm1d(self.start_ch * 4, momentum=self.BN_momentum),
            act,
            nn.Linear(self.start_ch * 4, self.start_ch * 4),
            nn.BatchNorm1d(self.start_ch * 4, momentum=self.BN_momentum),
            act,
            nn.Dropout(self.dropout_rate),
        )

        # ────────────── 업샘플 블록 ──────────────
        def UpBlock(in_ch, out_ch):
            return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),  # spatial ×2
                nn.BatchNorm2d(out_ch, momentum=self.BN_momentum),
                act,
                nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_ch, momentum=self.BN_momentum),
                # act,
                nn.Dropout2d(self.dropout_rate),
            )

        self.up1 = UpBlock(self.start_ch,       self.start_ch // 2)   # 2 → 4
        self.up2 = UpBlock(self.start_ch // 2,  self.start_ch // 4)   # 4 → 8
        self.up3 = UpBlock(self.start_ch // 4,  self.start_ch // 8)   # 8 → 16
        self.up4 = UpBlock(self.start_ch // 8,  self.start_ch // 16)  # 16 → 32  ← 추가

        # ────────────── 최종 1-채널 투영 ──────────────
        self.to_out = nn.Conv2d(self.start_ch // 16, 1, kernel_size=3, stride=1, padding=1)



    # ────────────── Forward ──────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seg1, seg2, seg3, seg4 = x[:, :7], x[:, 7:9], x[:, 9:14], x[:, 14:]

        emb1 = self.embed1(seg1)
        emb2 = self.embed2(seg2)
        emb3 = self.embed3(seg3)
        emb4 = self.embed4(seg4)
        
        # Method 1
        latent = emb1 + emb2 + emb3
        latent = torch.cat([latent, emb4], dim=1)
        
        # Method 2
        # latent = torch.cat([emb1, emb2, emb3, emb4], dim=1)  # (B, 512+512+512+512=2048)

        latent = self.fc(latent).view(-1, self.start_ch, 2, 2)    # (B, 2048, 2, 2)

        out = self.up1(latent)   # 4×4
        out = self.up2(out)      # 8×8
        out = self.up3(out)      # 16×16
        out = self.up4(out)      # 32×32
        out = self.to_out(out)   # (B, 1, 32, 32)

        return out[..., :31, :31]  # (B, 1, 31, 31)
    
    
    
    
if __name__ == "__main__": 
    model = SHCNN_(num_DV=17, dropout_rate=0.1, BN_momentum=0.1)
    summary = pytorch_model_summary.summary(model, torch.zeros(10, 15), show_input=True)
    print(summary)