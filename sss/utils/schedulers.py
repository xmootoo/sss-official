# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved. Taken from: https://github.com/facebookresearch/ijepa.

import math
from torch.optim.lr_scheduler import OneCycleLR


class PatchTSTSchedule(object):
    def __init__(self, optimizer, args, num_batches):
        self.optimizer = optimizer
        self.args = args
        self.epoch = 0
        
        if self.args.scheduler.lradj == "TST":
            self.one_cycle = OneCycleLR(optimizer = optimizer,
                                        steps_per_epoch = num_batches,
                                        pct_start = self.args.pct_start,
                                        epochs = self.args.sl.epochs,
                                        max_lr = self.args.sl.lr)

    def step(self):
        if self.args.scheduler.lradj == "type3":
            lr = self.args.sl.lr if self.epoch < 3 else self.args.sl.lr * (0.9 ** ((self.epoch - 3) // 1))
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
        elif self.args.scheduler.lradj == "TST":
            lrs = self.one_cycle.get_last_lr()
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group["lr"] = lrs[i]
        else:
            raise ValueError(f"Unsupported lr adjustment type: {self.args.scheduler.lradj}")
        
        self.epoch+=1



class WarmupCosineSchedule(object):

    def __init__(
        self,
        optimizer,
        warmup_steps,
        start_lr,
        ref_lr,
        T_max,
        last_epoch=-1,
        final_lr=0.
    ):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.T_max = T_max - warmup_steps
        self._step = 0.

    def step(self):
        self._step += 1
        if self._step < self.warmup_steps:
            progress = float(self._step) / float(max(1, self.warmup_steps))
            new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        else:
            # -- progress after warmup
            progress = float(self._step - self.warmup_steps) / float(max(1, self.T_max))
            new_lr = max(self.final_lr,
                         self.final_lr + (self.ref_lr - self.final_lr) * 0.5 * (1. + math.cos(math.pi * progress)))

        for group in self.optimizer.param_groups:
            group['lr'] = new_lr

        return new_lr


class CosineWDSchedule(object):

    def __init__(
        self,
        optimizer,
        ref_wd,
        T_max,
        final_wd=0.
    ):
        self.optimizer = optimizer
        self.ref_wd = ref_wd
        self.final_wd = final_wd
        self.T_max = T_max
        self._step = 0.

    def step(self):
        self._step += 1
        progress = self._step / self.T_max
        new_wd = self.final_wd + (self.ref_wd - self.final_wd) * 0.5 * (1. + math.cos(math.pi * progress))

        if self.final_wd <= self.ref_wd:
            new_wd = max(self.final_wd, new_wd)
        else:
            new_wd = min(self.final_wd, new_wd)

        for group in self.optimizer.param_groups:
            if ('WD_exclude' not in group) or not group['WD_exclude']:
                group['weight_decay'] = new_wd
        return new_wd
