import torchaudio
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from einops import rearrange


def pad_emb(emb, length):
    """emb has shape [1024, T]"""
    diff = length - emb.shape[-1]
    while diff > 0:
        pad = emb[:, :diff]
        emb = torch.concat([emb, pad], dim=-1)
        if emb.shape[-1] == length:
            return emb
        diff = length - emb.shape[-1]
    return emb


def clip_wav(mix_audio, clean_audio, length):
    """
    mix_audio and clean audio have shape [T]
    clip the audio to the certain length,
    """
    if mix_audio.size(0) < length:
        pad_tensor = torch.zeros(length - mix_audio.size(0))
        clean_audio = torch.cat([clean_audio, pad_tensor])
        mix_audio = torch.cat([mix_audio, pad_tensor])
    elif mix_audio.size(0) > length:
        mix_audio = mix_audio[:length]
        clean_audio = clean_audio[:length]
    return mix_audio, clean_audio


def clip_wav_tgt(mix_audio, tgt_audio, clean_audio, length):
    """
    mix_audio and clean audio have shape [T]
    clip the audio to the certain length,
    """
    if tgt_audio.size(0) > length:
        tgt_audio = tgt_audio[:length]
    elif tgt_audio.size(0) < length:
        pad_tensor = torch.zeros(length - tgt_audio.size(0))
        tgt_audio = torch.cat([tgt_audio, pad_tensor])

    if mix_audio.size(0) < length:
        pad_tensor = torch.zeros(length - mix_audio.size(0))
        clean_audio = torch.cat([clean_audio, pad_tensor])
        mix_audio = torch.cat([mix_audio, pad_tensor])
    elif mix_audio.size(0) > length:
        mix_audio = mix_audio[:length]
        clean_audio = clean_audio[:length]
    return mix_audio, tgt_audio, clean_audio


class WhamDataset(Dataset):
    def __init__(self, mix_path, source_path, length):
        self.mix = []
        self.source = []
        with open(mix_path, "r") as f:
            lines = f.readlines()
            for l in lines:
                self.mix.append(l.replace("\n", "").split(" ")[-1])
        with open(source_path, "r") as f:
            lines = f.readlines()
            for l in lines:
                self.source.append(l.replace("\n", "").split(" ")[-1])
        self.length = length

    def __len__(self):
        return len(self.mix)

    def __getitem__(self, idx):
        """
        return: mix_audio, clean_audio
        """
        mix_audio, _ = torchaudio.load(self.mix[idx])
        clean_audio, _ = torchaudio.load(self.source[idx])
        mix_audio = mix_audio[0]  # [T] reduce it to single-channel
        clean_audio = clean_audio[0]  # [T] reduce it to single-channel
        mix_audio, clean_audio = clip_wav(mix_audio, clean_audio, self.length)
        return (
            mix_audio,
            clean_audio,
        )  # reduce the multi-channel to single-channel audio by only considering the first channel


class LibriMixDataset(Dataset):
    def __init__(self, mix_path, source_path, length, mode="noise", emb=False):
        """
        Here the mix_length and source_path is the same, the source_path does not matter here due to the quality of the dataset.
        mode can be either noise or target
        """
        self.mode = mode
        print(f"Libri mode is {mode}")
        print(f"using embedding is {emb}")
        self.emb = emb
        self.mix = []
        self.source = []
        with open(mix_path, "r") as f:
            lines = f.readlines()
            for l in lines:
                self.mix.append(l.replace("\n", "").split(" ")[-1])
        with open(source_path, "r") as f:
            lines = f.readlines()
            for l in lines:
                self.source.append(l.replace("\n", "").split(" ")[-1])
        self.length = length
        pass

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        if self.emb:
            if self.mode == "noise":
                mix_emb = np.load(self.mix[idx])  # [1, 1024, T']
                clean_emb = np.load(self.source[idx])  # [1, 1024, T']
                mix_emb = mix_emb[:, :, : self.length]
                clean_emb = clean_emb[:, :, : self.length]
                mix_emb = torch.from_numpy(mix_emb[0])  # [1024, T']
                clean_emb = torch.from_numpy(clean_emb[0])  # [1024, T']
                mix_emb, clean_emb = pad_emb(mix_emb, self.length), pad_emb(
                    clean_emb, self.length
                )
                return mix_emb, clean_emb
            else:
                raise NotImplementedError("not implemented")
        else:
            if self.mode == "noise":
                mix_audio, _ = torchaudio.load(self.mix[idx])
                clean_audio, _ = torchaudio.load(self.source[idx])
                mix_audio = rearrange(mix_audio, "1 t -> t")
                clean_audio = rearrange(clean_audio, "1 t -> t")
                mix_audio, clean_audio = clip_wav(mix_audio, clean_audio, self.length)
                return mix_audio, clean_audio
            elif self.mode == "target":
                mix_audio, _ = torchaudio.load(self.mix[idx])
                clean_audio, _ = torchaudio.load(self.source[idx])
                tgt_audio, _ = torchaudio.load(random.choice(self.source))
                mix_audio = rearrange(mix_audio, "1 t -> t")
                clean_audio = rearrange(clean_audio, "1 t -> t")
                tgt_audio = rearrange(tgt_audio, "1 t -> t")[: self.length]
                return clip_wav_tgt(mix_audio, tgt_audio, clean_audio, self.length)


def load_dataset(config, mode="train"):
    if config["data"]["name"] == "WHAM":
        return load_wham_dataset(config, mode)
    elif config["data"]["name"] == "LibriMix":
        return load_librimix_dataset(config, mode)
    else:
        raise NotImplementedError("other dataset loading is not implemented")


def load_wham_dataset(config, mode="train"):
    """
    return training dataset and cv dataset
    """
    if mode == "train":
        tr_path = config["data"]["tr"]
        cv_path = config["data"]["cv"]
        return (
            WhamDataset(tr_path["mix"], tr_path["source"], config["data"]["length"]),
            WhamDataset(cv_path["mix"], cv_path["source"], config["data"]["length"]),
        )
    elif mode == "inference":
        infer_path = config["data"]["tt"]
        return WhamDataset(
            infer_path["mix"], infer_path["source"], config["data"]["length"]
        )


def load_librimix_dataset(config, mode="train"):
    if mode == "train":
        tr_path = config["data"]["tr"]
        cv_path = config["data"]["cv"]
        return (
            LibriMixDataset(
                tr_path["mix"],
                tr_path["source"],
                config["data"]["length"],
                mode=config["data"]["type"],
                emb=config["data"]["emb"],
            ),
            LibriMixDataset(
                cv_path["mix"],
                cv_path["source"],
                config["data"]["length"],
                mode=config["data"]["type"],
                emb=config["data"]["emb"],
            ),
        )
    elif mode == "inference":
        raise Exception("inference not implemented for loading librimix dataset")
    raise Exception("loading dataset not implemented")
