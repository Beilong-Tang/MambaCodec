import logging
import torch
import torch.nn as nn
import os
import time
from loss import pesq_fn, stoi_batch_fn
from .abs_trainer import AbsTrainer
import torch.distributed as dist


class CrossEntropyTrainer(AbsTrainer):
    def __init__(
        self, model, tr_data, cv_data, optim, config, args, device, loss_fn, *args_
    ):
        super().__init__(
            model, tr_data, cv_data, optim, config, args, device, loss_fn, *args_
        )

    def _train(self, loss_fn, optim, tr_data, epoch):
        self.model.train()
        start_time = time.time()
        total_loss = 0
        for batch, (mix_audio, clean_audio) in enumerate(tr_data):
            mix, clean = mix_audio.to(self.device), clean_audio.to(self.device)
            with torch.no_grad():
                true_code = self.model(clean, encode=True)  # [B,n_q,T]
            res = self.model(mix)  # [B,n_q,T,K]
            loss = loss_fn(res, true_code)
            loss.backward()

            #### gradient clipping
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(),1)
            optim.step()
            optim.zero_grad()
            total_loss += loss.item()
            if batch % self.log_interval == 0:
                with torch.no_grad():
                    true_audio = (
                        self.model(clean, skip_lm=True)
                        .audio_data.squeeze(1)
                        .cpu()
                        .numpy()
                    )  # [B,T]
                    output_audio = (
                        self.model(mix, recon=True)[1]
                        .audio_data.squeeze(1)
                        .cpu()
                        .numpy()
                    )  # [B,T]
                pesq = pesq_fn(output_audio, true_audio)
                stoi = stoi_batch_fn(output_audio, true_audio)
                loss, current = loss.item(), (batch + 1) * len(mix_audio)
                self._log(
                    f"epoch {epoch}, tr loss: {loss:>.7f}, pesq: {pesq:>.7f} , stoi: {stoi:>.7f}  [{current:>5d}/{(len(tr_data)*len(mix_audio)):>5d}], time: {(time.time() - start_time)*1000 :.2f}ms"
                )
                start_time = time.time()
        self.tr_loss[epoch] = total_loss / len(tr_data)

    def _eval(self, loss_fn, cv_data, epoch):
        self.model.eval()
        loss_dict = {}
        pesq_total = 0
        stoi_total = 0
        for mix_audio, clean_audio in cv_data:
            mix, clean = mix_audio.to(self.device), clean_audio.to(self.device)
            with torch.no_grad():
                true_audio = (
                    self.model(clean, skip_lm=True).audio_data.squeeze(1).cpu().numpy()
                )  # [B,T]
                output_audio = (
                    self.model(mix, recon=True)[1].audio_data.squeeze(1).cpu().numpy()
                )  # [B,T]
                pesq_total += pesq_fn(output_audio, true_audio)
                stoi_total += stoi_batch_fn(output_audio, true_audio)
        pesq_avg = pesq_total / len(cv_data)
        stoi_avg = stoi_total / len(cv_data)
        self._log(f"epoch {epoch}, cv, pesq: {pesq_avg:>.7f}, stoi: {stoi_avg:>.7f}")
        loss_dict["pesq"] = pesq_avg
        loss_dict["stoi"] = stoi_avg
        self.cv_loss[epoch] = loss_dict
