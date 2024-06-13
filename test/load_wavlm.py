import os 
os.environ['HTTP_PROXY'] = "http://proxy-dku.oit.duke.edu:3128"
os.environ['HTTPS_PROXY'] = "http://proxy-dku.oit.duke.edu:3128"

from transformers import AutoModel

import torch

model = AutoModel.from_pretrained("microsoft/wavlm-large")

audio = torch.randn(2, 16000*4)

output = model(audio, output_hidden_states = True)
print(output)

print(output.hidden_states[5].shape)