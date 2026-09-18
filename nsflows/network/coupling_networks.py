import torch
from torch import nn
    
class EquivariantTransformer(nn.Module):

    def __init__(self, input_size, output_size, device, conditioned=False, transformer_args={"depth" : 1, "dim" : 128}, n_freqs = 8):
        super(EquivariantTransformer, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.conditioned = conditioned

        self.n_freqs = n_freqs
        self.freqs = torch.arange(self.n_freqs, device=device).reshape(1, 1, -1) + 1

        if conditioned:
            self.lin_in = nn.Linear(self.input_size * 2 * self.n_freqs + 1, transformer_args["dim"])
        else:
            self.lin_in = nn.Linear(self.input_size * 2 * self.n_freqs, transformer_args["dim"])
        
        self.transformer_encoder = nn.TransformerEncoder(
                                        nn.TransformerEncoderLayer(
                                            d_model = transformer_args["dim"], 
                                            nhead = transformer_args["dim"]//64, 
                                            dim_feedforward = transformer_args["dim"]*4, 
                                            batch_first = True, 
                                            norm_first = True, 
                                            dropout=0.0
                                        ), 
                                        transformer_args["depth"]
                                    )
        
        self.lin_out = nn.Linear(transformer_args["dim"], self.output_size)


    def forward(self, x, condition=None):

        x = x.reshape(x.shape[0], -1, self.input_size)

        # Circular encoder (input must be between -1 and 1)
        cos_enc = torch.cos(self.freqs * torch.pi * x.unsqueeze(-1))
        sin_enc = torch.sin(self.freqs * torch.pi * x.unsqueeze(-1))

        x = torch.cat([
                cos_enc.view(x.shape[0], -1, self.input_size * self.n_freqs),
                sin_enc.view(x.shape[0], -1, self.input_size * self.n_freqs)
            ], dim=-1)
        if self.conditioned:
            x = torch.cat([x, condition.view(-1, 1, 1).repeat(1, x.shape[1], 1)], dim=-1)

        x = self.lin_in(x)
        x = self.transformer_encoder(x)
        x = self.lin_out(x)

        x = x.reshape(x.shape[0], -1)

        return x