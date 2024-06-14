import torch.nn as nn
import torch.nn.functional as F
import sys 
import logging 
import math
logger = logging.getLogger(__name__)

### add funcodec path
sys.path.append("/DKUdata/tangbl/FunCodec/funcodec/bin")
### 

from codec_inference import Speech2Token
from mamba_ssm import Mamba

import torch 
from torchaudio.models import Conformer
from einops import rearrange, repeat
def uniform_init(*shape: int):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t


def sample_vectors(samples, num: int):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


def kmeans(samples, num_clusters: int, num_iters: int = 10):
    """
    Kmeans for ecludian distance
    """
    dim, dtype = samples.shape[-1], samples.dtype

    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        diffs = rearrange(samples, "n d -> n () d") - rearrange(
            means, "c d -> () c d"
        )
        dists = -(diffs ** 2).sum(dim=-1) # [n,c]

        buckets = dists.max(dim=-1).indices # [n]
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1) #[c]

        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype) # [c, d]
        new_means.scatter_add_(0, repeat(buckets, "n -> n d", d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        means = torch.where(zero_mask[..., None], means, new_means)

    return means, buckets


def kmeans_batch(data, num_clusters: int, num_iters: int = 10, across_batch = True):
    """
    Run kmeans independently across batch
    data: [B, T, E]
    if across_batch is true, then the kmeans will be applied across batches (Not sure here)
    return: 
        means: [C, E] if across_batch is True or [B, C, E ] if across_batch is False
        clusters: [B, T] which cluster each embedding belong
    """
    if across_batch:
        shape = data.shape
        res = rearrange(data, "b t e -> (b t) e")
        means, res = kmeans(res, num_clusters, num_iters)
        res = res.view(*shape[:-1])
    else:
        means, res = [], []
        for d in data:
            result = kmeans(d, num_clusters, num_iters)
            means.append(result[0])
            res.append(result[1])
        res = rearrange(res, "b t -> b t").to(data.device)
        means = rearrange(means, "b t e -> b t e").to(data.device)
    return means, res 


class SelmCodec(nn.Module):
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
                bypass_quantizer = False, 
                sampling_rate = 16000,
                **kwargs
                ):
        super().__init__()
        ## true
        self.speech2Token = Speech2Token(config_path, model_path, device = device, bypass_quantizer = True, sampling_rate = sampling_rate)
        self.speech2TokenIndex = Speech2Token(config_path, model_path, device = device, bypass_quantizer = False, sampling_rate = sampling_rate)
        ## trainable
        self.codec = Speech2Token(config_path, model_path, device = device, bypass_quantizer =True, sampling_rate = sampling_rate)
        ## kmeans parameter
        self.kmeans_cluster = 300
        self.kmeans_iter = 10
        self.across_batch = True
        self.mambaModel = LanguageModel(emb_num = self.kmeans_cluster, emb_dim = 512)
        self.conformer = Conformer(input_dim=emb_dim, num_heads=4, ffn_dim= 256, num_layers= 4, depthwise_conv_kernel_size=3)
        self.register_buffer("lookup", None)
        for param in self.speech2Token.parameters():
            param.requires_grad = False
        for param in self.speech2TokenIndex.parameters():
            param.requires_grad = False
        mamba_param_num = sum(p.numel() for p in self.mambaModel.parameters())
        logger.info(f"mamba parameters {mamba_param_num}")
    
    def encode(self,x):
        """
        Args:
            x: input speech with shape (B, T)
        Returns:
            - embeddings after encoding with shape (B, T', emb_dim)
        """
        return self.codec.encode(x)
    
    def encode_true(self, x):
        """
        Args:
            x: input speech with shape (B, T)
        Returns:
            - embeddings after encoding with shape (B, T', emb_dim)
        """
        return self.speech2Token.encode(x)
    
    def tokenize(self, emb):
        """
        Args:
            emb: [B, T', emb_dim]
        Returns:
            tokens after running kmeans [B, T']
        Side effect: 
            Remember the kmeans center as the lookup table
        """
        means, tokens = kmeans_batch(emb, self.kmeans_cluster, self.kmeans_iter, self.across_batch)
        self.lookup = means
        return tokens
    
    def mamba(self, emb):
        """
        Args:
            emb: the token produced by the encoding process (B, T')
        Returns:
            - the probability of shape (B, T', C)
        """
        return self.mambaModel(emb)

    def detokenize(self, x):
        """
        args:
            x tokens with shape (B, T)
        Returns:
            reconstructed feature with shape (B, T, E)
        """
        if self.lookup == None:
            raise Exception("have to tokenize before this method!!!")
        if self.across_batch:
            res = F.embedding(x, self.lookup)
        else:
            res = [F.embedding(e, self.lookup[i]) for i,e in enumerate(x)]
            res = rearrange("b t e -> b t e")
        ### TODO: conformer model here
        res = self.conformer(res, torch.full((res.shape[0],),res.shape[1]).to(res.device)) # [B, T, E]
        return res[0]

    def decode_true(self, emb):
        """
        Args:
            emb: the embedding to be decoded
        Returns:
            - the reconstructed wav (B,  T'') (the wav might be a bit longer than the original one)
        """
        return self.speech2Token.decode_emb(emb).squeeze(1)
    
    @torch.no_grad()
    def inference(self, mix, clean):
        """
        inference mix and the clean (T)
        return output and clean [T']
        """
        true_emb = self.encode_true(clean) # (B, T', emb_dim)
        true_audio = self.decode_true(true_emb).squeeze(1) # [T']

        input_emb = self.encode(mix) # [B, T'. emb_dim]
        input_token = self.tokenize(input_emb) # [B,T']
        input_prob = self.mamba(input_token) # [B, T, C']
        input_prob_max = torch.argmax(input_prob, dim = -1) # [B, T']
        input_detokenize = self.detokenize(input_prob_max) # [B, T, E']
        output_audio = self.decode(input_detokenize).squeeze(1) # [T']
        return output_audio, true_audio

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

class LanguageModel(nn.Module):

    def __init__(self, emb_num, emb_dim = 512):
        """
        the language model mentioned in selm
        """
        super().__init__()
        self.audio_embedding = nn.Embedding(emb_num, emb_dim)
        self.position_encode = PositionalEncoding(emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=16, dim_feedforward = 1024, batch_first = True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=12)
        self.linear = nn.Linear(emb_dim, emb_num)

    def forward(self,x):
        """
            x, token with shape [B, T]
        Returns:
            y, probability with shape (B, T, C)
        """
        res = self.audio_embedding(x) # [B, T, emb_dim]
        res = self.position_encode(res) # [B, T, emb_dim]
        res = self.transformer_encoder(res) # [B,T, emb_dim]
        res = self.linear(res) # [B, T, C]
        res = torch.nn.functional.softmax(res, dim = -1)
        return res

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: shape [batch, T, emb_dim]
        Return:
            [batch, T, emb_dim] after applying positional encoding
        """
        x = rearrange(x, "b t e -> t b e")
        x = x + self.pe[:x.size(0)]
        x = rearrange(x, "t b e -> b t e")
        return self.dropout(x)
