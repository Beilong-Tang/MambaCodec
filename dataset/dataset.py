import torchaudio
import torch
from torch.utils.data import Dataset
from einops import rearrange


def clip_wav(mix_audio, clean_audio, length):
    """
    mix_audio and clean audio have shape [T]
    clip the audio to the certain length,
    """
    if mix_audio.size(0) <length:
        pad_tensor = torch.zeros(length - mix_audio.size(0))
        clean_audio = torch.cat([clean_audio, pad_tensor])
        mix_audio = torch.cat([mix_audio, pad_tensor])
    elif mix_audio.size(0) > length:
        mix_audio = mix_audio[:length]
        clean_audio = clean_audio[:length]
    return mix_audio, clean_audio

    pass

class WhamDataset(Dataset):
    def __init__(self, mix_path, source_path, length):
        self.mix = []
        self.source = [] 
        with open(mix_path,"r") as f:
            lines = f.readlines()
            for l in lines:
                self.mix.append(l.replace("\n","").split(" ")[-1])
        with open(source_path,"r") as f:
            lines = f.readlines()
            for l in lines:
                self.source.append(l.replace("\n","").split(" ")[-1])
        self.length = length
    
    def __len__(self):
        return len(self.mix)

    def __getitem__(self, idx):
        """
        return: mix_audio, clean_audio
        """
        mix_audio,_ = torchaudio.load(self.mix[idx])
        clean_audio, _ = torchaudio.load(self.source[idx])
        mix_audio = mix_audio[0] # [T] reduce it to single-channel
        clean_audio = clean_audio[0] # [T] reduce it to single-channel
        mix_audio, clean_audio = clip_wav(mix_audio, clean_audio, self.length)
        return mix_audio, clean_audio # reduce the multi-channel to single-channel audio by only considering the first channel

class LibriMix(Dataset):
    def __init__(self, mix_path, source_path, length, mode = "noise"):
        """
        Here the mix_length and source_path is the same, the source_path does not matter here due to the quality of the dataset.
        mode can be either noise or target
        """
        self.mix = []
        self.source = [] 
        with open(mix_path,"r") as f:
            lines = f.readlines()
            for l in lines:
                self.mix.append(l.replace("\n","").split(" ")[-1])
                self.source.append(l.replace("\n","").replace("s1","noise").split(" ")[-1])
        self.length = length
        
    def __len__(self):
        return len(self.mix)

    def __getitem__(self, idx):
        mixaudio, _ = torchaudio.load(self.mix[idx])
        cleanaudio, _ = torchaudio.load(self.source[idx])
        mix_audio = rearrange(mix_audio, "1 t -> t")
        clean_audio = rearrange(clean_audio, "1 t -> t")
        mix_audio, clean_audio = clip_wav(mix_audio, clean_audio, self.length)
        return mix_audio, clean_audio
    pass


def load_dataset(config, mode= "train"):
    if config['data']['name'] == 'WHAM':
        return load_wham_dataset(config, mode)
    elif config['data']['name'] == 'LibriMix':
        return load_librimix_dataset(config, mode)
    else:
        raise NotImplementedError("other dataset loading is not implemented")

def load_wham_dataset(config, mode = "train"):
    """
        return training dataset and cv dataset
    """
    if mode == "train":
        tr_path = config['data']['tr']
        cv_path = config['data']['cv']
        return (WhamDataset(tr_path['mix'], tr_path['source'], config['data']['length']),
                WhamDataset(cv_path['mix'], cv_path['source'], config['data']['length']))
    elif mode == "inference":
        infer_path = config['data']['tt']
        return WhamDataset(infer_path['mix'], infer_path['source'], config['data']['length'])

def load_librimix_dataset(config, mode = "train"):
    if mode == "train":


        pass
    elif mode == "inference":
        pass
    pass

