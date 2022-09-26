"""
encapsulation operations
"""

from .utils import cal_top_k_similar

from .dataset import preprocess_data, load_eval_data, DataController

from .lr_scheduler import poly_decay_lr, exp_decay_lr

from .skipgram import SkipGram

from .config import w2v_cfg
