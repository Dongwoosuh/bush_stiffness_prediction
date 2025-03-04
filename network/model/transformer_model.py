import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)

import pytorch_model_summary

__all__ = ["TransformerPredictor"]

class TransformerPredictor(nn.Module):
    def __init__(self, num_DV, embed_dim, num_heads, num_layers, output_dim, dropout=0.1):
        super(TransformerPredictor, self).__init__()
        
        self.input_mlp = nn.Sequential(
            nn.Linear(num_DV, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.query_embed = nn.Parameter(torch.randn(6,embed_dim))
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.outptut_mlp = nn.Linear(embed_dim, output_dim)
        
    
    def forward(self,x):
        x = self.input_mlp(x).unsqueeze(0)
        query = self.query_embed.unsqueeze(1).repeat(1, x.size(1), 1)
        
        x = self.transformer_encoder(x)
        
        x = self.transformer_decoder(query, x)
        
        x = self.outptut_mlp(x).squeeze(-1)
        
        return x.view(-1, 6, 16, 16)  # (batch, 6, 16, 16)
    

if __name__ == "__main__":
    model = TransformerPredictor(14, 128, 8, 3, 256)
    dummy_input = torch.randn(10,14)
    output = model(dummy_input)
    print(output.size())    