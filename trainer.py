import logging 
import torch
logger = logging.getLogger(__name__)


class Trainer():

    def __init__(self, model, tr_data, cv_data, optim, config, args, device):
        self.model = model 
        self.dataset = dataset 
        self.config = config 
        self.args = args
        self.epoch_start = 0
        self.tr_loss = []
        self.cv_loss = []
        self.optim = optim
        self.device = device
        self.train_codec = config['codec']['trainable']
        if not args.continue_from ==None:
            ckpt = torch.load(args.continue_from)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.epoch_start = ckpt['epoch']
            self.tr_loss = ckpt['tr_loss']
            self.cv_loss = ckpt['cv_loss']
            self.optim.load_state_dict(ckpt['optim'])
            pass
        pass

    
    def train():
        for epoch in range(self.epoch_start, config['epoch']):
            
            ### load dataset
            for mix_audio, clean_audio in tr_data:
                mix, clean = mix.to(device), clean.to(device)
                
                ## train with the codec first
                if not self.train_codec:
                    with torch.no_grad():
                        
                    pass

                pass




            pass


        pass
    
