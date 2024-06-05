import argparse
import yaml

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

from utils import get_instance
from trainer import Trainer

def main(args):
    ### prepare dataset
    

    ### prepare model

    ### start training loop
    pass

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required = True, type = str )
    parser.add_argument("--continue_from", type = str )
    parser.add_argument("--ddp", action = "store_true")
    parser.add_argument("--gpus", type = str, default = "4,5,6,7", required = True)
    parser.add_argument("--name", type = str, default = "base", required = True)
    parser.add_argument("--ckpt_path", type = str, required = True)
    args = parser.parse_args()
    main(args)
    pass