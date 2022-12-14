"""export checkpoint file into air, onnx, mindir models"""

import argparse

import numpy as np
from mindspore import Tensor, load_checkpoint, load_param_into_net, export, context

from src.config import w2v_cfg
from src.skipgram import SkipGram

parser = argparse.ArgumentParser(description='SkipGram export')
parser.add_argument("--device_id", type=int, default=0, help="device id")
parser.add_argument("--checkpoint_path", type=str, required=True, help="checkpoint file path.")
parser.add_argument("--file_name", type=str, default="skipgram", help="output file name.")
parser.add_argument('--file_format', type=str, choices=["AIR", "ONNX", "MINDIR"], default='MINDIR', help='file format')
parser.add_argument("--device_target", type=str, choices=["Ascend", "GPU"], default="Ascend", help="device target")

args = parser.parse_args()
context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
if args.device_target == "Ascend":
    context.set_context(device_id=args.device_id)

if __name__ == '__main__':
    param_dict = load_checkpoint(args.checkpoint_path)
    vocab_size = param_dict['c_emb.embedding_table'].shape[0]
    emb_size = param_dict['c_emb.embedding_table'].shape[1]
    net = SkipGram(vocab_size, emb_size)
    load_param_into_net(net, param_dict)
    center_words = Tensor(np.ones(w2v_cfg.batch_size, np.int32))
    pos_words = Tensor(np.ones(w2v_cfg.batch_size, np.int32))
    neg_words = Tensor(np.ones([w2v_cfg.batch_size, w2v_cfg.neg_sample_num], np.int32))
    export(net, center_words, pos_words, neg_words, file_name=args.file_name, file_format=args.file_format)
