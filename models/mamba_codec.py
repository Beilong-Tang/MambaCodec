import torch.nn as nn
import sys 
import logging 
logger = logging.getLogger(__name__)

### add funcodec path
sys.path.append("/DKUdata/tangbl/FunCodec/funcodec/bin")
### 

from codec_inference import Speech2Token
from mamba_ssm import Mamba

class MambaCodec(nn.Module):
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
                bypass_quantizer = True, 
                sampling_rate = 8000,
                **kwargs
                ):
        super().__init__()
        self.speech2Token = Speech2Token(config_path, model_path, device = device, bypass_quantizer =bypass_quantizer, sampling_rate = sampling_rate)

        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8)
        # transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12, d_model = 128)
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.mambaModel =transformer_encoder.to(device)
        
        # self.mambaModel =  MambaBlocks(
        #     # This module uses roughly 3 * expand * d_model^2 parameters
        #     num = mamba_num,
        #     d_model=d_model, # Model dimension d_model
        #     d_state=d_state,  # SSM state expansion factor
        #     d_conv=d_conv,    # Local convolution width
        #     expand=expand,    # Block expansion factor
        #     ).to(device)
        ## freeze model parameters
        for param in self.speech2Token.parameters():
            param.requires_grad = False
        mamba_param_num = sum(p.numel() for p in self.mambaModel.parameters())
        logger.info(f"mamba parameters {mamba_param_num}")
    
    def encode(self, x):
        """
        Args:
            x: input speech with shape (B, T)
        Returns:
            - embeddings after encoding with shape (B, T', emb_dim)
        """
        return self.speech2Token.encode(x)
    
    def mamba(self, emb):
        """
        Args:
            emb: the embedding produced by the encode process (B, T', emb_dim)
        Returns:
            - the embedding after mamba layers (B, T', emb_dim)
        """

        return self.mambaModel(emb)

    def decode(self, emb):
        """
        Args:
            emb: the embedding to be decoded
        Returns:
            - the reconstructed wav (B, T'') (the wav might be a bit longer than the original one)
        """
        return self.speech2Token.decode_emb(emb)

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

