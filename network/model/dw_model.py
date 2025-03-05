import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)

import pytorch_model_summary

__all__ = ['DWCNN_']

class DWCNN_(nn.Module):
    def __init__(self, 
                num_DV=17,
                BN_momentum=0.1, 
                dropout_rate=0.1, 
                start_ch=6*16*16,
                padding_param=0,
                kernel_size=3,
                stride=2,
                embdding_dim=128):
        super(DWCNN_, self).__init__()

        self.embdding_dim = embdding_dim
        self.start_ch = start_ch
        self.padding_param = padding_param
        self.kernel_size = kernel_size
        self.stride = stride
        self.dropout_rate = dropout_rate
        self.BN_momentum = BN_momentum

        seg1_dim = 8
        seg2_dim = 6
        seg3_dim = num_DV - 14

        self.embed1 = nn.Sequential(
            nn.Linear(seg1_dim, self.embdding_dim),
            nn.BatchNorm1d(self.embdding_dim, momentum=BN_momentum),
            nn.SiLU(inplace=True),
            # nn.Dropout(dropout_rate)
        )
        self.embed2 = nn.Sequential(
            nn.Linear(seg2_dim, self.embdding_dim),
            nn.BatchNorm1d(self.embdding_dim, momentum=BN_momentum),
            nn.SiLU(inplace=True),
            # nn.Dropout(dropout_rate)
        )
        self.embed3 = nn.Sequential(
            nn.Linear(seg3_dim, self.embdding_dim),
            nn.BatchNorm1d(self.embdding_dim, momentum=BN_momentum),
            nn.SiLU(inplace=True),
            # nn.Dropout(dropout_rate)
            )

        embed_total_dim =  embdding_dim * 3

        self.fc = nn.Sequential(
            nn.Linear(in_features=embed_total_dim, out_features=self.start_ch),
            nn.BatchNorm1d(self.start_ch, momentum=self.BN_momentum),
            nn.SiLU(inplace=True),
            nn.Dropout(self.dropout_rate)
        )

        # Autoencoder 방식의 Convolution 구조
        ch_list = [6, 16, 32, 64]

        # Encoder
        encoder_layers = []
        for i in range(len(ch_list) - 1):
            in_ch, out_ch = ch_list[i], ch_list[i + 1]
            encoder_layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=1),
                nn.BatchNorm2d(out_ch, momentum=BN_momentum),
                nn.SiLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        for i in range(len(ch_list) - 1, 0, -1):
            in_ch, out_ch = ch_list[i], ch_list[i - 1]
            decoder_layers.extend([
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=1, output_padding=1),
                nn.BatchNorm2d(out_ch, momentum=BN_momentum),
                nn.SiLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
        self.decoder = nn.Sequential(*decoder_layers)
            
    def forward(self, input):

        seg1 = input[:, :8]      
        seg2 = input[:, 8:14]      
        seg3 = input[:, 14:]   

        emb1 = self.embed1(seg1)   
        emb2 = self.embed2(seg2)   
        emb3 = self.embed3(seg3)

        x_embed = torch.cat([emb1, emb2, emb3], dim=1)  
        x = self.fc(x_embed)
        x = x.view(-1, 6, 16, 16)
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    

if __name__ == "__main__":
    model = DWCNN_()
    dummy_input = torch.randn(10,17)
    output = model(dummy_input)
    sumamry = pytorch_model_summary.summary(model, dummy_input, show_input=True)
    print(sumamry)