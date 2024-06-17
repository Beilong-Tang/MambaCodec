import argparse
import yaml
import sys
import os
import torch
import random
import numpy as np 
### set up logging
import logging 
import datetime

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import torch.distributed as dist

###
from torch.utils.data import DataLoader

from utils import get_instance, get_class, get_attr
from dataset.dataset import load_dataset
import loss

## detect error
torch.autograd.set_detect_anomaly(True)

# fix random seeds for reproducibility
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed_all(SEED)

## ddp process
def setup(rank, world_size, port_number):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port_number)

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def main(rank, world_size, args):
    ## set up logger
    now = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    log_dir = f"./logs/{args.name}"
    os.makedirs(log_dir, exist_ok = True)
    logging.basicConfig(format="%(asctime)s,%(name)s,%(levelname)s,%(message)s", 
                        datefmt= "%Y-%m-%d %H:%M:%S", filename = f"{log_dir}/{now}.log" , level= logging.INFO)
    logger = logging.getLogger(__name__)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s,%(name)s,%(levelname)s,%(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    print(f"rank {rank}")
    if rank ==0:
        logger.info(' '.join(sys.argv))

    setup(rank,world_size, args.port)
    # Set the CUDA device based on local_rank
    device = rank
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    ### prepare dataset
    tr_dataset, cv_dataset = load_dataset(config)
    tr_data = DataLoader(tr_dataset, batch_size = config['batch_size'],shuffle = False, sampler = DistributedSampler(dataset=tr_dataset))
    cv_data = DataLoader(cv_dataset, batch_size = config['batch_size'],shuffle = False, sampler = DistributedSampler(dataset=cv_dataset))
    ### prepare model
    model_class = get_class("models", config['model']['type'])
    model = model_class(**{**config['codec'], **config['model'], **{"device":device}}).to(device)
    model = DDP(model,device_ids=[rank])
    ### prepare optim
    optim = get_instance(torch.optim, config['optim'], model.parameters())
    ### start training loop
    trainer_class = get_class("trainer", f"{config['loss']}Trainer")
    trainer = trainer_class(model, tr_data, cv_data, optim, config, args, rank, get_attr(loss, config['loss'],"loss_fn"), logger)
    trainer.train()

    pass

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required = True, type = str )
    parser.add_argument("--continue_from", type = str )
    parser.add_argument("--ddp", action = "store_true")
    parser.add_argument("--gpus", type = str, default = "4,5,6,7")
    parser.add_argument("--device", type = str, default = "cuda:5")
    parser.add_argument("--name", type = str, required = True)
    parser.add_argument("--ckpt_path", type = str, required = True)
    parser.add_argument("--port", type = int, default = 12355)
    args = parser.parse_args()
    ###

    ### set up ddp 
    device_array = args.device.split(",") 
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in device_array])
    world_size = len(device_array)
    mp.spawn(main,
             args=(world_size, args),
             nprocs=world_size,
             join=True)
    pass