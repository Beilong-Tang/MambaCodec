import argparse
import yaml
import sys
import os
### set up logging
import logging 
import datetime
now = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
log_dir = "./logs"
os.makedirs(log_dir, exist_ok = True)
logging.basicConfig(format="%(asctime)s,%(name)s,%(levelname)s,%(message)s", 
                    datefmt= "%Y-%m-%d %H:%M:%S", filename = f"{log_dir}/{now}.log" , level= logging.INFO)
logger = logging.getLogger(__name__)
###
from torch.utils.data import DataLoader

from utils import get_instance
from trainer import Trainer
from dataset.dataset import load_dataset
from models.model import MambaCodec

def main(args):
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    ### prepare dataset
    tr_dataset, cv_dataset = load_dataset(config)
    tr_data = DataLoader(tr_dataset, batch_size = config['batch_size'])
    cv_data = DataLoader(cv_dataset, batch_size = config['batch_size'])
    ### prepare model
    model = MambaCodec(**config['codec'])
    ### prepare optim
    optim = get_instance(torch.optim, config['optim'], model.parameters())
    ### start training loop
    trainer = Trainer(model, tr_data, cv_data, optim, config, args, args.device)
    trainer.train()

    pass

if __name__ =="__main__":
    logger.info(' '.join(sys.argv))
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required = True, type = str )
    parser.add_argument("--continue_from", type = str )
    parser.add_argument("--ddp", action = "store_true")
    parser.add_argument("--gpus", type = str, default = "4,5,6,7")
    parser.add_argument("--device", type = str, default = "cuda:4")
    parser.add_argument("--name", type = str, default = "base", required = True)
    parser.add_argument("--ckpt_path", type = str, required = True)
    args = parser.parse_args()
    main(args)
    pass