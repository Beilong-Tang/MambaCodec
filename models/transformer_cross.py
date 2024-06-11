import torch.nn as nn
import torch
import sys 
import logging 
from einops import rearrange
logger = logging.getLogger(__name__)

### add funcodec path
sys.path.append("/DKUdata/tangbl/FunCodec/funcodec/bin")
### 

from codec_inference import Speech2Token
from mamba_ssm import Mamba

class TransformerCross(nn.Module):
    def __init__(self, 
                config_path, 
                model_path, 
                d_model,
                d_state,
                d_conv,
                expand,
                mamba_num,
                emb_dim = 128, ### Not sure about it yet
                device = "cpu", 
                hidden_dim =1024,
                nhead = 16, # number of attention
                bypass_quantizer = False, 
                sampling_rate = 16000,
                **kwargs
                ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.speech2Token = Speech2Token(config_path, model_path, device = device, bypass_quantizer =bypass_quantizer, sampling_rate = sampling_rate)
        self.device = device
        out_dim = 1024
        self.emb_dim = 128
        self.embedding_layers = nn.ModuleList([ nn.Embedding(out_dim, emb_dim).to(self.device) for _ in range(0,32)])
        self.linear_layers = nn.ModuleList([ nn.Linear(emb_dim, hidden_dim) for _ in range(0,32) ])
        # self.embed = nn.Embedding(out_dim, hidden_dim)
        # self.linear = nn.Linear(hidden_dim, out_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model= emb_dim, dim_feedforward = 512,  nhead=16, batch_first = True)
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.softmax = nn.Softmax(dim = -1)
        self.mambaModel =transformer_encoder.to(device)
        for param in self.speech2Token.parameters():
            param.requires_grad = False
        mamba_param_num = sum(p.numel() for p in self.mambaModel.parameters())
        logger.info(f"mamba parameters {mamba_param_num}")
    
    def encode(self, x):
        """
        Args:
            x: input speech with shape (B, T)
        Returns:
            - indexes after encoding with shape (n_q, B, T)
        """
        return self.speech2Token.encode_index(x)
    
    def mamba(self, emb):
        """
        Args:
            emb: the index produced by the encode process (n_q, B, T)
        Returns:
            - the possibility after mamba layers (B,T,n_q,K)
        """ 
        n_q, B, T = emb.shape
        res = torch.zeros(n_q, B, T, self.emb_dim).to(self.device) ### [n_q, B, T, H]
        for i, layer in enumerate(self.embedding_layers):
            res[i] = layer(emb[i])
        # res = self.embed(emb) # [n_q, B, T, H] weight (parameter), softmax
        res = res.sum(dim=0) #[B,T,emb]
        B, T, E = res.shape ## [n_q, B， T， H]
        res = self.mambaModel(res) # [B, T, E]
        result = [   l(res)   for l in self.linear_layers ] ## [n_q, B, T, K]
        result = rearrange(result, "n b t h -> n b t h")
        result = rearrange(result, "n b t h -> b n t h")
        result = self.softmax(result) 
        return result # [B,n_q, T, K ]

    def decode(self, emb):
        """
        Args:
            emb: the index to be decoded (B, T, n_q)
        Returns:
            - the reconstructed wav (B, T'') (the wav might be a bit longer than the original one)
        """
        return self.speech2Token.decode_index(emb).squeeze(1)

    pass

class MambaBlocks(nn.Module):
    def __init__(self, num, d_model, d_state, d_conv, expand):
        super().__init__()
        layers = []
        for _ in range(num):
            layers.append(Mamba(d_model=d_model, d_state = d_state, d_conv = d_conv, expand = expand))
        self.blocks = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args: input embedding with shape (B, T, E)
        Outputs: output with shape (B, T, E)

        """
        return self.blocks(x)
        pass

