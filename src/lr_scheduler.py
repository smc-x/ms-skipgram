"""Learning rate utilities."""

import numpy as np


def exp_decay_lr(learning_rate, decay_rate, decay_step, total_step, is_stair=False):
    """lr[i] = learning_rate∗pow(decay_rate, i / decay_step)
    """
    if is_stair:
        lrs = learning_rate * np.power(decay_rate, np.floor(np.arange(total_step) / decay_step))
    else:
        lrs = learning_rate * np.power(decay_rate, np.arange(total_step) / decay_step)
    return lrs.astype(np.float32)


def poly_decay_lr(learning_rate, end_learning_rate, decay_step, total_step, power, update_decay_step=False):
    """polynomial decay learning rate
    """
    lrs = []
    if update_decay_step:
        for step in range(total_step):
            tmp_decay_step = max(decay_step, decay_step * np.ceil(step / decay_step))
            lrs.append((learning_rate - end_learning_rate) * np.power(1 - step / tmp_decay_step, power)
                       + end_learning_rate)
    else:
        for step in range(total_step):
            step = min(step, decay_step)
            lrs.append((learning_rate - end_learning_rate) * np.power(1 - step / decay_step, power) + end_learning_rate)
    lrs = np.array(lrs)
    return lrs.astype(np.float32)
