import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)

import pytorch_model_summary

__all__ = ['CNN_small_dropout', 'MLPNN'] 
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
    
    
    
class BaseMLP(nn.Module):
    def __init__(self):
        super(BaseMLP, self).__init__()

    def get_activation(self, name):
        activations = {
            "SiLU": nn.SiLU(),
            "Sigmoid": nn.Sigmoid(),
            "Tanh": nn.Tanh(),
            "ELU": nn.ELU(),
            "LeakyReLU": nn.LeakyReLU(),
            "Mish": nn.Mish(),
            "SeLU": nn.SELU(),
            "ReLU": nn.ReLU(),
            "ReLU6": nn.ReLU6(),
            "None": nn.Identity(),
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
    
if __name__ == "__main__": 
    model = CNN_small_dropout(num_DV=12)
    input_tensor = torch.zeros(1, 12)  # (batch_size, in_features)
    model_summary = pytorch_model_summary.summary(model, input_tensor, show_input=True)
    print(model_summary)
