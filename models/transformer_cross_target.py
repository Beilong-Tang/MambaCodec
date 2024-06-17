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


## target speaker separation
class TransformerCrossTarget(nn.Module):
    def __init__(
        self,
        config_path,
        model_path,
        d_model,
        d_state,
        d_conv,
        expand,
        mamba_num,
        emb_dim=128,  ### Not sure about it yet
        device="cpu",
        hidden_dim=1024,
        nhead=16,  # number of attention
        bypass_quantizer=False,
        sampling_rate=16000,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.speech2Token = Speech2Token(
            config_path,
            model_path,
            device=device,
            bypass_quantizer=bypass_quantizer,
            sampling_rate=sampling_rate,
        )
        for param in self.speech2Token.parameters():
            param.requires_grad = False
        self.device = device
        out_dim = 1024
        self.emb_dim = 128
        self.hidden = 256
        self.embedding_layers = nn.ModuleList(
            [nn.Embedding(out_dim, self.hidden).to(self.device) for _ in range(0, 32)]
        )
        self.linear2 = nn.Linear(self.hidden, 1024)
        transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=128, dim_feedforward=512, nhead=16, batch_first=True
            ),
            num_layers=3,
        )
        transformer_encoder1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=128, dim_feedforward=512, nhead=16, batch_first=True
            ),
            num_layers=3,
        )
        self.inter = transformer_encoder.to(device)
        self.intra = transformer_encoder1.to(device)
        self.conv2d = nn.Conv2d(64, 128, (3, 3), padding=1)
        self.dconv2d = nn.ConvTranspose2d(128, 32, (3, 3), padding=(1, 1))
        mamba_param_num = sum(p.numel() for p in self.parameters())
        logger.info(f"total model parameters {mamba_param_num}")

    def encode(self, x):
        """
        Args:
            x: input speech with shape (B, T)
        Returns:
            - indexes after encoding with shape (n_q, B, T)
        """
        return self.speech2Token.encode_index(x)

    def mamba(self, mix, tgt):
        """
        Args:
            emb: the index produced by the encode process (n_q, B, T),
            tgt: (n_q, B, T)
        Returns:
            - the possibility after mamba layers ( B,n_q, T, 1024)
        """
        n_q, B, T = mix.shape
        ## embedding
        mix_emb = torch.zeros(n_q, B, T, self.hidden).to(
            self.device
        )  ### [n_q, B, T, e]
        for i, layer in enumerate(self.embedding_layers):
            mix_emb[i] = layer(mix[i])
        tgt_emb = torch.zeros(n_q, B, T, self.hidden).to(
            self.device
        )  ### [n_q, B, T, e]
        for i, layer in enumerate(self.embedding_layers):
            tgt_emb[i] = layer(tgt[i])
        mix_emb = rearrange(mix_emb, "n b t e -> b n t e")  # [B, n_q, T, N]
        tgt_emb = rearrange(tgt_emb, "n b t e -> b n t e")  # [B, n_q, T, N]
        ## concat the result
        res = torch.cat((mix_emb, tgt_emb), dim=1)  # [B, 2 *n_q, T, N]
        res = self.conv2d(res)  # [B, C, T, N]
        B, C, T, N = res.shape
        res = rearrange(res, "b c t n -> (b t) n c")  # [B *T, N, C ]
        res = self.inter(res)  # [B*T, N C]
        res = res.view(B, C, T, N)  # [B, C, T, N]
        res = rearrange(res, "b c t n -> (b n) t c")  # [B *N, T, C]
        res = self.intra(res)  # [B *N, T, C]
        res = res.view(B, C, T, N)  # [B,C,T,N]
        res = self.dconv2d(res)  # [B, 32, T, N]
        res = self.linear2(res)  # [B, 32, T, 1024]
        return res

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
            layers.append(
                Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            )
        self.blocks = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args: input embedding with shape (B, T, E)
        Outputs: output with shape (B, T, E)

        """
        return self.blocks(x)
        pass
