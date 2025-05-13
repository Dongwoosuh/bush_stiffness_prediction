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
    def __init__(self, 
                num_DV=17,
                dropout_rate=0.1, 
                BN_momentum = 0.1,
                start_ch = 2048,
                embedding_dim1=1024,
                embedding_dim2=1024,
                # embedding_dim3=128,
                activation='SiLU'
                ):
        super(SHCNN_, self).__init__()

        self.padding_param = 0
        self.kernel_size = 3
        self.stride = 1
        self.embedding_dim1 = embedding_dim1
        self.embedding_dim2 = embedding_dim2
        self.start_ch = start_ch 
        self.dropout_rate = dropout_rate
        self.BN_momentum = BN_momentum
        
        seg1_dim = 8
        seg2_dim = 6
        seg3_dim = num_DV - 14

        self.embed1 = nn.Sequential(
            nn.Linear(seg1_dim, self.embedding_dim1),
            nn.BatchNorm1d(self.embedding_dim1, momentum=self.BN_momentum),
            self.get_activation(activation),
            nn.Linear(self.embedding_dim1, self.embedding_dim1),
            # nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        self.embed2 = nn.Sequential(
            nn.Linear(seg2_dim, self.embedding_dim2),
            nn.BatchNorm1d(self.embedding_dim2, momentum=self.BN_momentum),
            self.get_activation(activation),
            nn.Linear(self.embedding_dim2, self.embedding_dim2),
            nn.Dropout(dropout_rate)
        )
        self.embed3 = nn.Sequential(
            nn.Linear(seg3_dim, self.embedding_dim1),
            nn.BatchNorm1d(self.embedding_dim1, momentum=self.BN_momentum),
            self.get_activation(activation),
            nn.Dropout(dropout_rate)
            )

        embed_total_dim =  self.embedding_dim1 + self.embedding_dim2

        self.fc = nn.Sequential(
            nn.Linear(in_features=embed_total_dim, out_features=self.start_ch * 2 * 2),
            nn.BatchNorm1d(self.start_ch * 2 * 2, momentum=self.BN_momentum),
            self.get_activation(activation),
            nn.Linear(self.start_ch * 2 * 2, self.start_ch * 2 * 2),
            nn.Dropout(self.dropout_rate)
            
        )

        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(self.start_ch, self.start_ch // 2, kernel_size=self.kernel_size, 
                                 stride=self.stride, padding=self.padding_param),
            nn.BatchNorm2d(self.start_ch // 2, momentum=self.BN_momentum),
            self.get_activation(activation),
            nn.ConvTranspose2d(self.start_ch // 2, self.start_ch // 2, kernel_size=self.kernel_size, 
                                 stride= self.stride, padding=self.padding_param),
            nn.BatchNorm2d(self.start_ch // 2, momentum=self.BN_momentum),
            self.get_activation(activation),
            nn.AvgPool2d(3, stride=1, padding=0, count_include_pad=False),
            nn.Dropout(dropout_rate),

            nn.ConvTranspose2d(self.start_ch // 2, self.start_ch // 4, kernel_size=self.kernel_size, 
                                 stride=self.stride, padding=self.padding_param),
            nn.BatchNorm2d(self.start_ch // 4, momentum=self.BN_momentum),
            self.get_activation(activation),
            nn.ConvTranspose2d(self.start_ch // 4, self.start_ch // 4, kernel_size=self.kernel_size, 
                                 stride=self.stride, padding=self.padding_param),
            nn.BatchNorm2d(self.start_ch // 4, momentum=self.BN_momentum),
            self.get_activation(activation),
            nn.AvgPool2d(3, stride=1, padding=0, count_include_pad=False),
            nn.Dropout(dropout_rate),

            nn.ConvTranspose2d(self.start_ch // 4, self.start_ch // 8, kernel_size=self.kernel_size, 
                                 stride=self.stride, padding=self.padding_param),
            nn.BatchNorm2d(self.start_ch // 8, momentum=self.BN_momentum),
            self.get_activation(activation),
            nn.ConvTranspose2d(self.start_ch // 8, self.start_ch // 8, kernel_size=self.kernel_size, 
                                 stride=self.stride, padding=self.padding_param),
            nn.BatchNorm2d(self.start_ch // 8, momentum=self.BN_momentum),
            self.get_activation(activation),
            nn.AvgPool2d(3, stride=1, padding=0, count_include_pad=False),
            nn.Dropout(dropout_rate),

            nn.ConvTranspose2d(self.start_ch // 8, self.start_ch // 16, kernel_size=3, 
                                 stride=self.stride, padding=0),
            nn.BatchNorm2d(self.start_ch // 16, momentum=self.BN_momentum),
            self.get_activation(activation),
            nn.ConvTranspose2d(self.start_ch // 16, self.start_ch // 16, kernel_size=3, 
                                 stride=self.stride, padding=0),
            nn.BatchNorm2d(self.start_ch // 16, momentum=self.BN_momentum),
            self.get_activation(activation),
            nn.AvgPool2d(3, stride=1, padding=0, count_include_pad=False),
            nn.Dropout(dropout_rate),

            nn.ConvTranspose2d(self.start_ch // 16, self.start_ch // 32, kernel_size=3, 
                                 stride=self.stride, padding=0),
            nn.BatchNorm2d(self.start_ch // 32, momentum=self.BN_momentum),
            self.get_activation(activation),
            nn.ConvTranspose2d(self.start_ch // 32, self.start_ch // 32, kernel_size=3, 
                                 stride=self.stride, padding=0),
            nn.BatchNorm2d(self.start_ch // 32, momentum=self.BN_momentum),
            self.get_activation(activation),
            nn.AvgPool2d(3, stride=1, padding=1,count_include_pad=False),
            nn.Dropout(dropout_rate),
        )

        self.conv_last = nn.Sequential(
            nn.ConvTranspose2d(self.start_ch // 32, 6, kernel_size=3, stride=self.stride, padding=0),
            nn.Flatten(),
            nn.Linear(in_features=6 * 16 * 16, out_features=6 * 16 * 16),
        )

    def forward(self, input):

        seg1 = input[:, :8]      
        seg2 = input[:, 8:14]      
        # seg3 = input[:, 14:]   

        emb1 = self.embed1(seg1)   
        emb2 = self.embed2(seg2)   
        # emb3 = self.embed3(seg3)

        x_embed = torch.cat([emb1, emb2], dim=1)  

        # x_embed = emb1 + emb2 + emb3
        x = self.fc(x_embed)  
        x = x.view(-1, self.start_ch, 2, 2)
        x = self.conv5(x)
        x = self.conv_last(x).view(-1, 6, 16, 16)
        return x
    
    
    
    
    
if __name__ == "__main__": 
    model = SHCNN_(num_DV=17, dropout_rate=0.1, BN_momentum=0.1)
    summary = pytorch_model_summary.summary(model, torch.zeros(10, 17), show_input=True)
    print(summary)