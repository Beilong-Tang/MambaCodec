import torch.nn as nn
import torch
import sys
import logging
from einops import rearrange
import dac
from audiotools import AudioSignal

logger = logging.getLogger(__name__)

### add funcodec path
sys.path.append("/DKUdata/tangbl/FunCodec/funcodec/bin")
###

from codec_inference import Speech2Token
from mamba_ssm import Mamba


class TransformerCross(nn.Module):
    def __init__(
        self,
        config_path,
        model_path,
        d_model,
        d_state,
        d_conv,
        expand,
        mamba_num,
        emb_dim=1024,  ### Not sure about it yet
        device="cpu",
        hidden_dim=1024,
        nhead=16,  # number of attention
        bypass_quantizer=False,
        sampling_rate=16000,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.codec = dac.DAC.load(
            dac.utils.download(model_type="16khz")
        )  # 12 code books
        self.device = device
        out_dim = 256
        self.emb_dim = emb_dim
        self.embedding_layers = nn.ModuleList(
            [nn.Embedding(1024, out_dim).to(self.device) for _ in range(0, 12)]
        )
        self.linear_layers = nn.ModuleList(
            [nn.Linear(out_dim, emb_dim).to(self.device) for _ in range(0, 12)]
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_dim, dim_feedforward=2048, nhead=16, batch_first=True
        )
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.softmax = nn.Softmax(dim=-1)
        self.mambaModel = transformer_encoder.to(device)
        self.weights = nn.Parameter(torch.randn(12))
        for param in self.codec.parameters():
            param.requires_grad = False
        mamba_param_num = sum(p.numel() for p in self.parameters())
        logger.info(f"mamba parameters {mamba_param_num}")

    def mamba(self, emb):
        """
        Args:
            emb: the index produced by the encode process (B, n_q, T)
        Returns:
            - the possibility after mamba layers (B,n_q,T,K)
        """
        B, n_q, T = emb.shape
        emb = rearrange(emb, "b n t -> n b t")  # [n_q, B, T]
        res = torch.zeros(n_q, B, T, self.emb_dim).to(self.device)  ### [n_q, B, T, H]
        for i, layer in enumerate(self.embedding_layers):
            res[i] = layer(emb[i])
        weights = rearrange(self.weights, "b -> b 1 1 1")
        res = torch.sum(weights * res, dim=0)  # [weighted sum ] -> [B, T, emb]
        B, T, E = res.shape  ## [B， T， H]
        res = self.mambaModel(res)  # [B, T, E]
        result = [l(res) for l in self.linear_layers]  ## [n_q, B, T, K]
        result = rearrange(result, "n b t h -> n b t h")
        result = rearrange(result, "n b t h -> b n t h")
        result = self.softmax(result)
        return result  # [B,n_q, T, K ]

    def forward(self, x, encode=False, recon=False, sample_rate=16000, skip_lm=False):
        """if encode is true, only use the encoder to produce indexes
            x : audio signal with shape [B, T]
            recon: if true, decode the signal as well, else, just return the code
            if skip_lm, then just return the reconstructed audio
        Returns:
            if encode, only return the codes index [B,n_q, T]
            else
                the possibility of output codes [B, n_q, T, K]
                the reconstruced audio if recon is True
        """
        res = x.unsqueeze(1)  # [B, 1, T']
        audio_sig = AudioSignal(res, sample_rate)
        compress_audio = self.codec.compress(audio_sig)  # compressed_audio
        codes = compress_audio.codes  # [B, n_q, T]
        res = self.mamba(codes)  # [B,n_q, T, H]
        if encode:
            return codes  # [B, n_q, T]
        if skip_lm:
            return self.codec.decompress(compress_audio)
        if not recon:
            return res  # AudioData(B,1,T)
        else:
            new_codes = torch.argmax(res, dim=-1)  # [B, n_q, T]
            compress_audio.codes = new_codes
            return res, self.codec.decompress(
                compress_audio
            )  # [B,n_q, T, K], AudioData(B, 1, T)

    pass
