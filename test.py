import sys 

### add funcodec path
sys.path.append("/DKUdata/tangbl/FunCodec/funcodec/bin")

from codec_inference import Speech2Token

bypass_quantizer = True

sample_rate = 8000

device = "cuda:4"

model_path = "/DKUdata/tangbl/FunCodec/egs/LibriTTS/codec/exp/audio_codec-encodec-zh_en-general-16k-nq32ds640-pytorch/model.pth"

config_path = "/DKUdata/tangbl/FunCodec/egs/LibriTTS/codec/exp/audio_codec-encodec-zh_en-general-16k-nq32ds640-pytorch/config.yaml"

speech2Token = Speech2Token(config_path, model_path, device = device, bypass_quantizer =bypass_quantizer, sampling_rate = sample_rate)

import torch
from mamba_ssm import Mamba

audio = torch.randn(2, 32000).to(device)

emb = speech2Token.encode(audio)

print(f"emb shape {emb.shape}")

batch, length, dim = emb.shape
model = Mamba(
    # This module uses roughly 3 * expand * d_model^2 parameters
    d_model=dim, # Model dimension d_model
    d_state=16,  # SSM state expansion factor
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
).to(device)
y = model(emb)

print(f"shape of y {y.shape}")

assert y.shape == emb.shape

output = speech2Token.decode_emb(y)

print(output.shape)

