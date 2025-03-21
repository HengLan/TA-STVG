# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from bisect import bisect_right
import torch
from torch.optim import Optimizer


# separating MultiStepLR with WarmupLR
# but the current LRScheduler design doesn't allow it
class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of increasing integers."
                "Got {}".format(milestones),
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


class WarmupPolyLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        gamma,
        max_iter,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.max_iter = max_iter
        self.gamma = gamma     # The poly power
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupPolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        assert self.last_epoch >= 0
        return [
            base_lr
            * warmup_factor
            * ((1 - self.last_epoch / self.max_iter) ** self.gamma)  
            for base_lr in self.base_lrs
        ]


class WarmupReduceLROnPlateau(object):
    def __init__(
        self,
        optimizer,
        gamma=0.5,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
        patience=2,
        threshold=1e-4,
        cooldown=1,
        logger=None,
    ):
        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        self.patience = patience
        self.threshold = threshold
        self.cooldown = cooldown
        self.stage_count = 0
        self.best = -1e12
        self.num_bad_epochs = 0
        self.under_cooldown = self.cooldown
        self.logger = logger

        # The following code is copied from Pytorch=1.2.0
        # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
            last_epoch = 0
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.last_epoch = last_epoch

        self.step(last_epoch)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self):
        warmup_factor = 1
        # during warming up
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        # 
        return [
            base_lr
            * warmup_factor
            * self.gamma ** self.stage_count
            for base_lr in self.base_lrs
        ]

    def step(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        # The following part is modified from ReduceLROnPlateau
        if metrics is None:
            # not conduct validation yet
            pass
        else:
            if float(metrics) > (self.best + self.threshold):
                self.best = float(metrics)
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1
            
            if self.under_cooldown > 0:
                self.under_cooldown -= 1
                self.num_bad_epochs = 0

            if self.num_bad_epochs >= self.patience:
                if self.logger is not None:
                    self.logger.info("Trigger Schedule Decay, RL has been reduced by factor {}".format(self.gamma))
                self.stage_count += 1  # this will automatically decay the learning rate
                self.under_cooldown = self.cooldown
                self.num_bad_epochs = 0


        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


def adjust_learning_rate(
    cfg,
    optimizer,
    curr_step: int,
    num_training_steps: int
):
    """
    Adjust the lr according to the schedule.
    """
    num_warmup_steps = round(cfg.SOLVER.WARMUP_PROP * num_training_steps)
    iter_per_epoch = round(num_training_steps / cfg.SOLVER.MAX_EPOCH)
    now_epoch = curr_step // iter_per_epoch
    
    drop_step = cfg.SOLVER.SCHEDULE.DROP_STEP

    if cfg.SOLVER.SCHEDULE.TYPE == "multistep_with_warmup":
        gamma = 0.1 ** bisect_right(drop_step, now_epoch)
        if curr_step < num_warmup_steps:
            text_encoder_gamma = float(curr_step) / float(max(1, num_warmup_steps))
        else:
            text_encoder_gamma = max(
                0.0,
                float(num_training_steps - curr_step)
                / float(max(1, num_training_steps - num_warmup_steps)),
            )
        temp_decoder_gamma = text_encoder_gamma
    elif cfg.SOLVER.SCHEDULE.TYPE == "multistep_with_warmup_all":
        if curr_step < num_warmup_steps:
            gamma = float(curr_step) / float(max(1, num_warmup_steps))
        else:
            gamma = 0.1 ** bisect_right(drop_step, now_epoch)
        text_encoder_gamma = gamma
        temp_decoder_gamma = text_encoder_gamma
    else:
        raise ValueError(f"Unsupported Schedule Type : {cfg.SOLVER.SCHEDULE.TYPE}")

    base_lrs = [cfg.SOLVER.BASE_LR, cfg.SOLVER.VIS_BACKBONE_LR, cfg.SOLVER.TEXT_LR, cfg.SOLVER.TEMP_LR, cfg.SOLVER.VERB_LR]
    gammas = [gamma, gamma, text_encoder_gamma, temp_decoder_gamma, gamma]
    assert len(optimizer.param_groups) == len(base_lrs)
    for param_group, lr, gamma_group in zip(optimizer.param_groups, base_lrs, gammas):
        param_group["lr"] = lr * gamma_group