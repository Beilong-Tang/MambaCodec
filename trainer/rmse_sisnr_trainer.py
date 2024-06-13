import logging 
import torch
import torch.nn as nn 
import os 
import time
from loss import si_snr_loss_fn, mse_loss_fn
from abs_trainer import AbsTrainer
import torch.distributed as dist
logger = logging.getLogger(__name__)


class RmseSisnrTrainer(AbsTrainer):
    def __init__(self, model, tr_data, cv_data, optim, config, args, device, loss_fn):
        super().__init__(model, tr_data, cv_data, optim, config, args, device, loss_fn)
        ##
        ## loss ration
        ##
        self.loss_ratio = config['ratio']
    
    def _train(self, loss_fn, optim, tr_data, epoch):
        self.model.train()
        start_time = time.time()
        for batch, (mix_audio, clean_audio) in enumerate(tr_data):
            mix, clean = mix_audio.to(self.device), clean_audio.to(self.device)
            ## true
            true_emb = self.model.encode(clean)
            true_audio = self.model.decode(true_emb)
            
            ## mix
            input_emb = self.model.encode(mix)
            output_y = self.model.mamba(input_emb)
            output_audio = self.model.decode(output_y)
            loss = loss_fn(output_y,true_emb, output_audio, true_audio)
            loss.backward()
            
            #### gradient clipping
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(),1)
            optim.step()
            optim.zero_grad()
            self.tr_loss[epoch] = loss.item()
            if batch % self.log_interval == 0:
                true_audio = self.model.decode(true_emb)
                output_audio = self.model.decode(output_y)
                si_snr_loss = si_snr_loss_fn(output_audio, true_audio).item()
                loss, current = loss.item(), (batch + 1) * len(mix_audio)
                logger.info(f"epoch {epoch}, tr loss: {loss:>.7f}, si snr loss: {si_snr_loss:>.7f}  [{current:>5d}/{(len(tr_data)*len(mix_audio)):>5d}], time: {(time.time() - start_time)*1000 :.2f}ms")
                start_time = time.time()
    
    def _eval(self, loss_fn, cv_data, epoch):
        self.model.eval()
        loss_dict = {}
        mse_loss_total = 0 
        si_snr_loss_total = 0 
        for mix_audio, clean_audio in cv_data:
            mix, clean = mix_audio.to(self.device), clean_audio.to(self.device)
            with torch.no_grad():
                input_emb = self.model.encode(mix)
                output_y = self.model.mamba(input_emb)
                output_audio = self.model.decode(output_y)
                true_emb = self.model.encode(clean)
                true_audio = self.model.decode(true_emb)
                mse_loss = loss_fn(output_y, true_emb, output_audio, true_audio).item()
                si_snr_loss = si_snr_loss_fn(output_audio, true_audio).item()
                si_snr_loss_total += si_snr_loss
                mse_loss_total += mse_loss
        mse_loss_avg= mse_loss_total / len(cv_data)
        si_snr_loss_avg = si_snr_loss_total / len(cv_data)
        logger.info(f"epoch {epoch}, cv loss: {(mse_loss_avg) :>.7f}, si_snr loss: {(si_snr_loss_avg) :>7f}")
        loss_dict['loss'] = mse_loss_avg
        loss_dict['si_snr'] = si_snr_loss_avg
        self.cv_loss[epoch] = loss_dict
    
