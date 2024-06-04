import torchaudio
from torch.utils.data import Dataset


class WhamDataset(Dataset):
    def __init__(self, mix_path, source_path):
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
    
    def __len__(self):
        return len(self.mix)

    def __getitem__(self, idx):
        """
        return: mix_audio, clean_audio
        """
        mix_audio,_ = torchaudio.load(self.mix[idx]) 
        clean_audio, _ = torchaudio.load(self.source[idx])
        return mix_audio[0], clean_audio[0] # reduce the multi-channel to single-channel audio by only considering the first channel


def load_dataset(config):
    if config['data']['name'] == 'WHAM':
        return load_wham_dataset(config)
    else:
        raise NotImplementedError("other dataset loading is not implemented")

def load_wham_dataset(config):
    """
        return training dataset and cv dataset
    """
    tr_path = config['data']['tr']
    cv_path = config['data']['cv']
    return WhamDataset(tr_path['mix'], tr_path['source']), WhamDataset(cv_path['mix'], cv_path['source'])

