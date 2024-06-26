import logging
import torch
import torch.nn as nn
import os
import time
from einops import rearrange
from loss import si_snr_loss_fn
import torch.distributed as dist


class AbsTrainer:

    def __init__(
        self,
        model,
        tr_data,
        cv_data,
        optim,
        config,
        args,
        device,
        loss_fn,
        logger,
        *args_,
        **kwargs,
    ):
        self.model = model.module
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
        self.log_interval = config["log_interval"]
        self.loss_fn = loss_fn
        self.logger = logger
        os.makedirs(os.path.join(self.ckpt_path, self.name), exist_ok=True)
        if not args.continue_from == None:
            self._log(f"loading model from {args.continue_from}...")
            ckpt = torch.load(args.continue_from)
            self.model.module.load_state_dict(ckpt["model_state_dict"])
            self.epoch_start = ckpt["epoch"]
            self.tr_loss = ckpt["tr_loss"]
            self.cv_loss = ckpt["cv_loss"]
            self.optim.load_state_dict(ckpt["optim"])
            pass
        pass

    def _save(self, model, tr_loss, cv_loss, epoch, optim, path):
        if self.device == 0:
            print(f"saving model... for epoch {epoch}")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optim": optim.state_dict(),
                    "tr_loss": tr_loss,
                    "cv_loss": cv_loss,
                },
                path,
            )
        pass

    def _log(self, msg):
        if self.device == 0:
            self.logger.info(msg)
        pass

    def train(self):
        for epoch in range(self.epoch_start, self.config["epoch"]):
            self._log(f"...epoch {epoch}...")
            self.tr_data.sampler.set_epoch(epoch)
            ### training
            self._train(self.loss_fn, self.optim, self.tr_data, epoch)
            ### evaluation calculate mse loss as well as si_snr loss
            self._eval(self.loss_fn, self.cv_data, epoch)
            ### save model
            self._save(
                self.model,
                self.tr_loss,
                self.cv_loss,
                epoch,
                self.optim,
                os.path.join(self.ckpt_path, self.name, f"epoch{epoch}.pth"),
            )
            dist.barrier()

    pass
