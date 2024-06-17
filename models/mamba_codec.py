import torch.nn as nn
import torch
import sys 
import logging 
logger = logging.getLogger(__name__)
import dac 
from audiotools import AudioSignal 


class MambaCodec(nn.Module):
    def __init__(self, 
                config_path, 
                model_path, 
                d_model,
                d_state,
                d_conv,
                expand,
                mamba_num,
                emb_dim = 1024, ### Not sure about it yet
                device = "cpu", 
                bypass_quantizer = False, 
                sampling_rate = 8000,
                **kwargs
                ):
        super().__init__()
        self.codec = dac.DAC.load(dac.utils.download(model_type="16khz")) # 12 code books
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=8, batch_first = True)
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.mambaModel = transformer_encoder.to(device)
        ## freeze model parameters
        for param in self.codec.parameters():
            param.requires_grad = False
        mamba_param_num = sum(p.numel() for p in self.parameters())
        logger.info(f"mamba parameters {mamba_param_num}")
    
    def encode(self, x):
        """
        Args:
            x: input speech with shape (B, T)
        Returns:
            - embeddings after encoding with shape (B, T', emb_dim)
        """
        x = x.unsqueeze(1) # [B, 1, T]
        return self.codec.encode(x)[0].permute(0,2,1) # [B, T', emb_dim]
    
    def mamba(self, emb):
        """
        Args:
            emb: the embedding produced by the encode process (B, T, emb_dim)
        Returns:
            - the embedding after mamba layers (B, T', emb_dim)
        """
        # emb = emb.permute(0, 2, 1) # [B， T‘， emb_dim]
        return self.mambaModel(emb) # [B, T', emb_dim]

    def decode(self, emb):
        """
        Args:
            emb: the embedding to be decoded [B, T',emb_dim]
        Returns:
            - the reconstructed wav (B,   T'') (the wav might be a bit longer than the original one)
        """
        emb = emb.permute(0, 2, 1) # [B, emb_dim, T']
        return self.codec.decode(emb).squeeze(1)
    
    @torch.no_grad()
    def inference(self, mix, clean):
        """
        inference mix and the clean (T)
        return output and clean [T']
        """
        raise NotImplementedError("not implemented")
        # true_emb = self.encode(clean)
        # true_audio = self.decode(true_emb)

        # input_emb = self.encode(mix)
        # output_y = self.mamba(input_emb)
        # output_audio = self.decode(output_y)
        # return output_audio, true_audio

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

