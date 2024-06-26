### inference method ###
########################

import argparse
import os
import torch
import torchaudio
import tqdm
import yaml

from torch.utils.data import Subset
from utils import get_class
from dataset.dataset import load_dataset


def main(args):
    output_path = os.path.join(args.output_path, args.name)
    os.makedirs(output_path, exist_ok=True)
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    tr_dataset, cv_dataset = load_dataset(config)

    ckpt = torch.load(args.ckpt_path)
    model_class = get_class("models", config["model"]["type"])
    model = model_class(
        **{**config["codec"], **config["model"], **{"device": args.device}}
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(args.device)

    subset_indices = list(range(args.num))
    subset = Subset(tr_dataset, subset_indices)
    with torch.no_grad():
        for idx, (mix, clean) in tqdm.tqdm(enumerate(subset), total=args.num):
            mix, clean = mix.to(args.device).unsqueeze(0), clean.to(
                args.device
            ).unsqueeze(0)
            audio, true_audio = model.inference(mix, clean)
            torchaudio.save(
                os.path.join(output_path, f"{idx}_true.wav"),
                true_audio.cpu(),
                config["model"]["sampling_rate"],
            )
            torchaudio.save(
                os.path.join(output_path, f"{idx}_output.wav"),
                audio.cpu(),
                config["model"]["sampling_rate"],
            )
            torchaudio.save(
                os.path.join(output_path, f"{idx}_mix.wav"),
                mix.cpu(),
                config["model"]["sampling_rate"],
            )
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True, type=str)
    parser.add_argument("--device", type=str, default="cuda:5")
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num", type=int, default=10)
    args = parser.parse_args()
    main(args)
    pass
