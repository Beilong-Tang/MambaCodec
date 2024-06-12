import logging 
import torch
import torch.nn as nn 
import os 
import time
from loss import si_snr_loss_fn
logger = logging.getLogger(__name__)


class SelmTrainer():
    def __init__(self, model, tr_data, cv_data, optim, config, args, device, loss_fn):
        self.model = model 
        self.tr_data = tr_data
        self.cv_data = cv_data
        self.config = config 
        self.ckpt_path = args.ckpt_path
        self.name = args.name
        self.epoch_start = 0
        self.tr_loss = {}
        self.cv_loss = {}
        self.optim = optim
        self.device = device
        self.log_interval = config['log_interval']
        self.loss_fn = loss_fn
        self.kl_div_loss_fn = loss_fn()[0]
        self.mse_loss_fn = loss_fn()[1]
        os.makedirs(os.path.join(self.ckpt_path, self.name), exist_ok = True)
        if not args.continue_from ==None:
            logger.info(f"loading model from {args.continue_from}...")
            ckpt = torch.load(args.continue_from)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.epoch_start = ckpt['epoch']
            self.tr_loss = ckpt['tr_loss']
            self.cv_loss = ckpt['cv_loss']
            self.optim.load_state_dict(ckpt['optim'])
            pass
        pass
    
    def _save(self, model, tr_loss, cv_loss, epoch, optim, path):
        torch.save({'epoch':epoch, 
                    'model_state_dict':model.state_dict(),
                    'optim': optim.state_dict(),
                    'tr_loss':tr_loss,
                    'cv_loss':cv_loss }, path)
        pass
    
    def _train(self, loss_fn, optim, tr_data, epoch):
        self.model.train()
        start_time = time.time()
        for batch, (mix_audio, clean_audio) in enumerate(tr_data):
            mix, clean = mix_audio.to(self.device), clean_audio.to(self.device)
            ## true
            self.model.eval()
            with torch.no_grad():
                true_emb = self.model.encode_true(clean)
                true_token = self.model.tokenize(true_emb) # [B, T]
                true_audio = self.model.decode_true(true_emb).squeeze(1)
            optim.zero_grad()
            self.model.train()
            ## mix
            input_emb = self.model.encode(mix)
            ## train encoder
            mse_loss= self.mse_loss_fn(input_emb, true_emb)
            mse_loss.backward()
            optim.step()
            optim.zero_grad()
            input_token = self.model.tokenize(input_emb) #[B,T]
            input_prob = self.model.mamba(input_token) # [B,T, C]
            input_prob_max = torch.argmax(input_prob, dim = -1) #[B,T]
            input_detokenize = self.model.detokenize(input_prob_max) #[B, T, E]
            output_audio = self.model.decode_true(input_detokenize).squeeze(1) # output audio [B,T]
            ### multi-task learning
            # 1. kl_div loss
            kl_div_loss = self.kl_div_loss_fn(input_prob, true_token)
            kl_div_loss.backward()
            optim.step()
            optim.zero_grad()
            # 2. mse loss
            mse_loss = self.mse_loss_fn(input_detokenize, true_emb)
            mse_loss.backward()
            optim.step()
            optim.zero_grad()
            self.tr_loss[epoch] = {"kl_div": kl_div_loss.item(), "mse": mse_loss.item()}
            if batch % self.log_interval == 0:
                si_snr_loss = si_snr_loss_fn(output_audio, true_audio).item()
                loss, current = si_snr_loss, (batch + 1) * len(mix_audio)
                logger.info(f"epoch {epoch}, tr kl loss: {kl_div_loss.item():>.7f}, mse loss {mse_loss.item():>.7f} si snr loss: {loss:>.7f}  [{current:>5d}/{(len(tr_data)*len(mix_audio)):>5d}], time: {(time.time() - start_time)*1000 :.2f}ms")
                start_time = time.time()
    
    def _eval(self, loss_fn, cv_data, epoch):
        self.model.eval()
        loss_dict = {}
        mse_loss_total = 0 
        mse_loss_de_total = 0
        si_snr_loss_total = 0
        kl_div_loss_total = 0
        with torch.no_grad():
            true_emb = self.model.encode_true(clean)
            true_token = self.model.tokenize(true_emb) # [B, T]
            true_audio = self.model.decode_true(true_emb).squeeze(1)
            self.model.train()
            ## mix
            input_emb = self.model.encode(mix)
            ## train encoder
            mse_loss_total += self.mse_loss_fn(input_emb, true_emb).item()
            input_token = self.model.tokenize(input_emb) #[B,T]
            input_prob = self.model.mamba(input_token) # [B,T, C]
            input_prob_max = torch.argmax(input_prob, dim = -1) #[B,T]
            input_detokenize = self.model.detokenize(input_prob_max) #[B, T, E]
            output_audio = self.model.decode_true(input_detokenize).squeeze(1) # output audio [B,T]
            si_snr_loss_total += si_snr_loss_fn(output_audio, true_audio)
            ### multi-task learning
            # 1. kl_div loss
            kl_div_loss_total  += self.kl_div_loss_fn(input_prob, true_token).item()
            # 2. mse loss
            mse_loss_de_total += self.mse_loss_fn(input_detokenize, true_emb).item()
        mse_loss_avg= mse_loss_total / len(cv_data)
        mse_loss_de_avg = mse_loss_de_total / len(cv_data)
        si_snr_loss_avg = si_snr_loss_total / len(cv_data)
        kl_div_loss_avg  = kl_div_loss_total / len(cv_data)
        logger.info("cross validation....")
        logger.info(f"epoch {epoch}, cv kl loss: {kl_div_loss_avg:>.7f}, mse loss {mse_loss_avg:>.7f}, mse de loss {mse_loss_de_avg :>.7f} si snr loss: {si_snr_loss_avg:>.7f}")
        loss_dict['mse'] = mse_loss_avg
        loss_dict['mse_de'] = mse_loss_de_avg
        loss_dict['si_snr'] = si_snr_loss_avg
        loss_dict['kl_div'] = kl_div_loss_avg
        self.cv_loss[epoch] = loss_dict
    
    def train(self):
        for epoch in range(self.epoch_start, self.config['epoch']):
            logger.info(f"...epoch {epoch}...")
            ### training 
            self._train(self.loss_fn, self.optim, self.tr_data, epoch)
            ### evaluation calculate mse loss as well as si_snr loss
            self._eval(self.loss_fn, self.cv_data, epoch)
            ### save model 
            self._save(self.model, self.tr_loss, self.cv_loss, epoch, self.optim, os.path.join(self.ckpt_path, self.name, f"epoch{epoch}.pth"))
