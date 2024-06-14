import logging 
import torch
import torch.nn as nn 
import os 
import time
from loss import si_snr_loss_fn
import torch.distributed as dist
from .abs_trainer import AbsTrainer


class SelmTrainer(AbsTrainer):
    def __init__(self, model, tr_data, cv_data, optim, config, args, device, loss_fn,*args_):
        super().__init__(model, tr_data, cv_data, optim, config, args, device, loss_fn,*args_)
        self.kl_div_loss_fn = loss_fn()[0]
        self.mse_loss_fn = loss_fn()[1]
    
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
            # mse_loss.backward()
            # optim.step()
            # optim.zero_grad()
            self.tr_loss[epoch] = {"kl_div": kl_div_loss.item(), "mse": mse_loss.item()}
            if batch % self.log_interval == 0:
                si_snr_loss = si_snr_loss_fn(output_audio, true_audio).item()
                loss, current = si_snr_loss, (batch + 1) * len(mix_audio)
                self._log(f"epoch {epoch}, tr kl loss: {kl_div_loss.item():>.7f}, mse loss {mse_loss.item():>.7f} si snr loss: {loss:>.7f}  [{current:>5d}/{(len(tr_data)*len(mix_audio)):>5d}], time: {(time.time() - start_time)*1000 :.2f}ms")
                start_time = time.time()
    
    def _eval(self, loss_fn, cv_data, epoch):
        self.model.eval()
        loss_dict = {}
        mse_loss_total = 0 
        mse_loss_de_total = 0
        si_snr_loss_total = 0
        kl_div_loss_total = 0
        with torch.no_grad():
            for batch, (mix_audio, clean_audio) in enumerate(cv_data):
                mix, clean = mix_audio.to(self.device), clean_audio.to(self.device)
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
        self._log("cross validation....")
        self._log(f"epoch {epoch}, cv kl loss: {kl_div_loss_avg:>.7f}, mse loss {mse_loss_avg:>.7f}, mse de loss {mse_loss_de_avg :>.7f} si snr loss: {si_snr_loss_avg:>.7f}")
        loss_dict['mse'] = mse_loss_avg
        loss_dict['mse_de'] = mse_loss_de_avg
        loss_dict['si_snr'] = si_snr_loss_avg
        loss_dict['kl_div'] = kl_div_loss_avg
        self.cv_loss[epoch] = loss_dict
    