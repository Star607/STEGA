import argparse
import os
from math import ceil
from pathlib import Path
import socket
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import helpers
from tqdm import tqdm
from model import MyTransformerConfig, MyTransformer

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int, choices=[0, 1, 2, 3, 4])
    parser.add_argument("--cuda", default="0", type=str)
    parser.add_argument("--dtype", default="float16", type=str)
    parser.add_argument("--method", default="STEGA", type=str)
    parser.add_argument(
        "--data",
        type=str,
        default="BJ_Taxi",
        choices=["BJ_Taxi", "Porto_Taxi", "Shanghai_Taxi", "Chengdu_Taxi"],
    )
    parser.add_argument("--datapath", default="", type=str)
    parser.add_argument("--out_dir", default="out", type=str)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--vocab_size", default=0, type=int)

    # gnn settings
    parser.add_argument("--embed_dim", default=256, type=int)
    parser.add_argument("--gps_emb_dim", default=10, type=int)
    parser.add_argument("--gnn_layer_num", default=2, type=int)
    parser.add_argument("--gnn_head_num", default=2, type=int)
    parser.add_argument("--gnn", default="gat", type=str)

    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lr_patience", type=float, default=2)
    parser.add_argument("--lr_decay_ratio", type=float, default=1e-2)
    parser.add_argument("--early_stop_lr", type=float, default=1e-6)
    parser.add_argument("--batch_size", default=32, type=int)

    # transformer settings
    parser.add_argument("--t_embed_dim", default=10, type=int)
    parser.add_argument("--tf_head_num", default=2, type=int)
    parser.add_argument("--tf_layer_num", default=2, type=int)
    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--bias", default=False, type=bool)

    # optimization settings
    parser.add_argument("--grad_clip", default=1.0, type=float)
    parser.add_argument("--eval_only", default=False, type=bool)
    parser.add_argument("--eval_interval", default="10", type=int)

    args = parser.parse_args()

    helpers.set_random_seed(args.seed)
    args.hostname = socket.gethostname()
    args.datapath = f"./data/{args.data}"
    device = torch.device(
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

    # set log path
    log_dir = f"./logs"
    log_prefix = (
        f"{args.method}-{args.data}-{train_str}-train-{args.hostname}-gpu{args.cuda}"
    )
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = helpers.set_logger(log_dir=log_dir, log_prefix=log_prefix)
    logger.info(args)

    # set saved path
    os.makedirs(args.out_dir, exist_ok=True)
    path_model_best = f"{args.out_dir}/{args.data}_{train_str}_ckpt_best.pth"
    path_model_last = f"{args.out_dir}/{args.data}_{train_str}_ckpt_last.pth"

    # load data
    adj_mx = helpers.read_adjcent_file(args.data)
    adj_dense = adj_mx.toarray()

    adj_no_isolate_file = f"./data/{args.data}/adjacent_mx_fill.npz"
    if os.path.exists(adj_no_isolate_file):
        adj_dense = np.load(adj_no_isolate_file)
    else:
        for i in range(len(adj_dense)):
            adj_dense[i][i] = 1
            if adj_dense[i].sum() == 0:
                adj_dense[np.random.randint(0, len(adj_dense), 1)[0]] = 1
        np.save(adj_no_isolate_file, adj_dense)

    dist_geo = np.load(f"./data/{args.data}/dist_geo.npy")  
    node_features, vocab_size = helpers.read_node_feature_file(args.data, device)
    args.vocab_size = vocab_size

    map_manager = helpers.MapManager(dataset_name=args.data)
    data_feature = {
        "adj_mx": adj_mx,
        "node_features": node_features,
        "img_width": map_manager.img_width,
        "img_height": map_manager.img_height,
    }
    gnn_config = {
        "gnn_model": args.gnn,
        "embed_dim": args.embed_dim,
        "no_gps_emb": True,
        "gps_emb_dim": args.gps_emb_dim,
        "num_of_layers": args.gnn_layer_num,
        "num_of_heads": args.gnn_head_num,
        "concat": False,
        "distance_mode": "l2",
    }
    tf_config = {
        "n_embd": args.embed_dim
        + args.t_embed_dim,  # args.embed_dim includes node feature and gps_embd
        "t_embd": args.t_embed_dim,
        "block_size": map_manager.block_size,
        "n_head": args.tf_head_num,
        "n_layer": args.tf_layer_num,
        "dropout": args.dropout,
        "bias": args.bias,
    }

    # load model
    model_args = dict(
        gnn_config=gnn_config,
        data_feature=data_feature,
        tf_config=tf_config,
        seed=args.seed,
        data=args.data,
        datapath=args.datapath,
        vocab_size=args.vocab_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
    )
    model = MyTransformer(MyTransformerConfig(**model_args)).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="max",
        patience=args.lr_patience,
        factor=args.lr_decay_ratio,
    )

    best_val_avg_acc_top1 = 0
    start_i = 0
    for epoch in range(args.epochs):
        model.train()
        train_total_loss = 0
        tra_loader, val_loader = helpers.generate_data_loader(
            args.data,
            "tra_and_val",
            args.batch_size,
            adj_dense,
            dist_geo,
            device,
        )
        for batch in tqdm(tra_loader, desc="train transformer"):
            with args.ctx:
                optimizer.zero_grad(set_to_none=True)
                _, _, loss = model(
                    batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]
                )
                loss.backward()
                train_total_loss += loss.item()
                if args.grad_clip != 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            start_i = start_i + args.batch_size
        logger.info(
            "epoch {}: train_loss {:.6f}".format(
                epoch, train_total_loss / len(tra_loader)
            )
        )
        if (epoch + 1) % args.eval_interval == 0:
            model.eval()
            val_hit, val_cnt = 0, 0
            start_i = 0
            for batch in tqdm(tra_loader, desc="valid transformer"):
                with args.ctx:
                    logits_masked, _, _ = model(
                        batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]
                    )
                value, index = torch.topk(logits_masked, 1, dim=-1)
                val_hit += (index.squeeze(-1) == batch[1]).sum()
                val_cnt += (batch[1] != -1).sum()
                start_i = start_i + args.batch_size
            avg_ac = val_hit / val_cnt
            logger.info("epoch {}: eval_top1_ac {:.6f}".format(epoch, avg_ac))
            if avg_ac > best_val_avg_acc_top1:
                best_val_avg_acc_top1 = avg_ac
                ckpt = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "epoch": epoch,
                    "best_val_avg_acc_top1": best_val_avg_acc_top1,
                }
                logger.info(f"saving checkpoint to {args.out_dir}")
                torch.save(ckpt, path_model_best)
            model.train()
        # torch.cuda.empty_cache()

    # save model
    ckpt_last = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_args": model_args,
        "epoch": epoch,
        "best_val_avg_acc_top1": avg_ac,
    }
    torch.save(ckpt_last, path_model_last)
