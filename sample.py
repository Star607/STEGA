import argparse
import json
import os
import socket
import sys

import helpers
import numpy as np
import torch

from model import MyTransformer, MyTransformerConfig
from torch.profiler import ProfilerActivity, profile, record_function
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=0, type=int, choices=[0, 1, 2, 3, 4])
parser.add_argument("--cuda", default="0", type=str)
parser.add_argument("--dtype", default="float16", type=str)
parser.add_argument("--method", default="STEGA", type=str)
parser.add_argument("--load_type", default="last", choices=["best", "last"])
parser.add_argument(
    "--data", type=str, default="BJ_Taxi", choices=["BJ_Taxi", "Porto_Taxi", 'Shanghai_Taxi', 'Chengdu_Taxi']
)
parser.add_argument("--out_dir", default="out", type=str)
parser.add_argument("--num_samples", default=5000, type=int)
parser.add_argument("--top_k", default=5, type=int)
parser.add_argument("--init_from", default="resume", type=str)
parser.add_argument("--temperature", default="0.8", type=float)
parser.add_argument("--debug", default=False, type=helpers.str2bool)

# hyperparameter
parser.add_argument("--tf_head_num", default=2, type=int)
parser.add_argument("--tf_layer_num", default=2, type=int)
parser.add_argument("--gnn", default="gat", type=str)
parser.add_argument("--gnn_layer_num", default=2, type=int)
parser.add_argument("--gnn_head_num", default=2, type=int)
args = parser.parse_args()

helpers.set_random_seed(args.seed)
args.hostname = socket.gethostname()
args.datapath = f"./data/{args.data}"

args.device = torch.device(
    f"cuda:{args.cuda}" if torch.cuda.is_available() else torch.device("cpu")
)
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[args.dtype]
args.ctx = torch.autocast(device_type="cuda", dtype=ptdtype)

args.out_dir = args.out_dir + "/main"

train_str = f"{args.seed}-{args.gnn}-gly{args.gnn_layer_num}-ghd{args.gnn_head_num}-tly{args.tf_layer_num}-thd{args.tf_head_num}"

path_model_best = f"{args.out_dir}/{args.data}_{train_str}_ckpt_best.pth"
path_model_last = f"{args.out_dir}/{args.data}_{train_str}_ckpt_last.pth"
path_test_gene = f"{args.out_dir}/{args.data}_{train_str}_gene_{args.num_samples}.txt"

log_dir = f"./logs"
log_prefix = (
    f"{args.method}-{args.data}-{train_str}-sample-{args.hostname}-gpu{args.cuda}"
)
logger = helpers.set_logger(log_dir=log_dir, log_prefix=log_prefix)
logger.info(args)

# load model
map_manager = helpers.MapManager(dataset_name=args.data)
if args.init_from == "resume":
    if args.load_type == "best":
        logger.info(f"load test model from {path_model_best}")
        ckpt = torch.load(path_model_best, map_location=args.device)
    else:
        logger.info(f"load test model from {path_model_last}")
        ckpt = torch.load(path_model_last, map_location=args.device)
    modelConfig = MyTransformerConfig(**ckpt["model_args"])
    modelConfig.device = args.device

    model = MyTransformer(modelConfig)
    state_dict = ckpt["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
else:
    logger.info("wrong assignment!")

model.eval()
model.to(args.device)

adj_mx = helpers.read_adjcent_file(args.data)
adj_dense = adj_mx.toarray()
dist_geo = np.load(f"./data/{args.data}/dist_geo.npy")

# generate trajectories
with torch.no_grad():
    with args.ctx:
        pred_data = model.generate(
            args,
            dist_geo,
            adj_dense,
            args.num_samples,
            temperature=args.temperature,
            top_k=args.top_k,
        )
with open(path_test_gene, "w") as f:
    json.dump(pred_data, f)
