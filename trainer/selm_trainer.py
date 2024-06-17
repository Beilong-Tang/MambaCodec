import logging
import torch
import torch.nn as nn
import os
import time

# from loss import si_snr_loss_fn
import torch.distributed as dist
from .abs_trainer import AbsTrainer
from loss import pesq_fn, stoi_batch_fn


class SelmTrainer(AbsTrainer):
    def __init__(
        self, model, tr_data, cv_data, optim, config, args, device, loss_fn, *args_
    ):
        super().__init__(
            model, tr_data, cv_data, optim, config, args, device, loss_fn, *args_
        )
        self.kl_div_loss_fn = loss_fn()[0]
        self.mse_loss_fn = loss_fn()[1]
        self.optim_lm = torch.optim.Adam(model.module.mambaModel.parameters(), lr=1e-3)
        self.optim_conf = torch.optim.Adam(model.module.conformer.parameters(), lr=1e-3)

    def _train(self, loss_fn, optim, tr_data, epoch):
        optim_lm = self.optim_lm
        optim_conf = self.optim_conf
        self.model.train()
        start_time = time.time()
        for batch, (mix_audio, clean_audio) in enumerate(tr_data):
            mix, clean = mix_audio.to(self.device), clean_audio.to(
                self.device
            )  # [B, E, T]
            mix = mix.permute(0, 2, 1)
            clean = clean.permute(0, 2, 1)  # [B, T, E]
            ## true
            with torch.no_grad():
                true_token = self.model.tokenize(clean)  # [B, T]
            input_token = self.model.tokenize(mix)  # [B,T]
            input_prob = self.model.mamba(input_token)  # [B,T, C]
            kl_loss = self.kl_div_loss_fn(input_prob, true_token)
            kl_loss.backward()
            optim_lm.step()
            optim_lm.zero_grad()
            input_prob_max = torch.argmax(input_prob, dim=-1)  # [B,T]
            input_detokenize = self.model.detokenize(input_prob_max)  # [B, T, E]

            mse_loss = self.mse_loss_fn(input_detokenize, clean)
            mse_loss.backward()
            optim_conf.step()
            optim_conf.zero_grad()
            self.tr_loss[epoch] = {"kl_div": kl_loss.item(), "mse": mse_loss.item()}
            if batch % self.log_interval == 0:
                with torch.no_grad():
                    output_audio = (
                        self.model.decode(input_detokenize).detach().cpu().numpy()
                    )  # output audio [B,T]
                    true_audio = (
                        self.model.decode(clean).detach().cpu().numpy()
                    )  # [B,T]
                pesq = pesq_fn(output_audio, true_audio)
                stoi = stoi_batch_fn(output_audio, true_audio)
                current = (batch + 1) * len(mix_audio)
                self._log(
                    f"epoch {epoch}, tr kl loss: {kl_loss.item():>.7f}, mse loss: {mse_loss.item():>.7f}, pesq: {pesq:>.7f} , stoi: {stoi:>.7f}  [{current:>5d}/{(len(tr_data)*len(mix_audio)):>5d}], time: {(time.time() - start_time)*1000 :.2f}ms"
                )
                start_time = time.time()

    def _eval(self, loss_fn, cv_data, epoch):
        self._log("cross validation....")
        self.model.eval()
        loss_dict = {}
        mse_loss_total = 0
        kl_div_loss_total = 0
        pesq_total = 0
        stoi_total = 0
        with torch.no_grad():
            for batch, (mix_audio, clean_audio) in enumerate(cv_data):
                mix, clean = mix_audio.to(self.device), clean_audio.to(self.device)
                mix = mix.permute(0, 2, 1)  # [B, T, E]
                clean = clean.permute(0, 2, 1)  # [B, T, E]
                true_token = self.model.tokenize(true_emb)  # [B, T]
                true_audio = self.model.decode(clean).detach().cpu().numpy()  # [B, T]
                input_token = self.model.tokenize(mix)  # [B,T]
                input_prob = self.model.mamba(input_token)  # [B,T, C]
                kl_div_loss_total += self.kl_div_loss_fn(input_prob, true_token).item()
                input_prob_max = torch.argmax(input_prob, dim=-1)  # [B,T]
                input_detokenize = self.model.detokenize(input_prob_max)  # [B, T, E]
                mse_loss_total += self.mse_loss_fn(input_detokenize, clean).item()
                output_audio = (
                    self.model.decode(input_detokenize).detach().cpu().numpy()
                )
                pesq_total += pesq_fn(output_audio, true_audio)
                stoi_total += stoi_batch_fn(output_audio, true_audio)
        mse_loss_avg = mse_loss_total / len(cv_data)
        kl_div_loss_avg = kl_div_loss_total / len(cv_data)
        pesq_avg = pesq_total / len(cv_data)
        stoi_avg = stoi_total / len(cv_data)
        self._log(
            f"epoch {epoch}, cv kl loss: {kl_div_loss_avg:>.7f}, mse loss: {mse_loss_avg:>.7f}, pesq: {pesq_avg:>.7f} , stoi: {stoi_avg:>.7f}..."
        )
        loss_dict["mse"] = mse_loss_avg
        loss_dict["pesq"] = pesq_avg
        loss_dict["stoi"] = stoi_avg
        loss_dict["kl_div"] = kl_div_loss_avg
        self.cv_loss[epoch] = loss_dict
