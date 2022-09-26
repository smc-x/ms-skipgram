"""Config parameters for skipgram models."""

import os

class ConfigSkipgram:
    """
    Config parameters for the Skipgram.

    Examples:
        ConfigSkipgram()
    """
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    par_dir = os.path.dirname(cur_dir)

    lr = 1e-3                          # initial learning rate
    end_lr = 1e-4                      # end learning rate
    train_epoch = 1                    # training epoch
    data_epoch = 10                    # generate data epoch
    power = 1                          # decay rate of learning rate
    batch_size = 128                   # batch size
    dataset_sink_mode = True
    print_interval = 1000
    emb_size = 288                     # embedding size
    min_count = 5                      # keep vocabulary that have appeared at least 'min_count' times
    window_size = 5                    # window size of center word
    neg_sample_num = 5                 # number of negative words in negative sampling
    save_checkpoint_steps = int(5e5)   # step interval between two checkpoints
    keep_checkpoint_max = 15                                    # maximal number of checkpoint files
    temp_dir = os.path.join(par_dir, 'temp/')                   # save files generated during code execution
    ckpt_dir = os.path.join(par_dir, 'temp/ckpts/')             # directory that save checkpoint files
    ms_dir = os.path.join(par_dir, 'temp/ms_dir/')              # directory that saves mindrecord data
    w2v_emb_save_dir = os.path.join(par_dir, 'temp/w2v_emb/')   # directory that saves word2vec embeddings
    train_data_dir = os.path.join(par_dir, 'data/train_data/')  # directory of training corpus
    eval_data_dir = os.path.join(par_dir, 'data/eval_data/')    # directory of evaluating data

w2v_cfg = ConfigSkipgram()
