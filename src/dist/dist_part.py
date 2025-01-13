import argparse
import socket
import time
import dgl
import dgl.nn.pytorch as dglnn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import datetime
import json
from torch.nn.parallel import DistributedDataParallel as DDP
from part import *

def write_json(NAME,rank,subGSavePath):
    int32_size = 4
    DataPath = subGSavePath + f"/part{rank}/raw_G.bin"
    NodePath = subGSavePath + f"/part{rank}/raw_nodes.bin"
    edge_size = os.path.getsize(DataPath)
    edge_num = edge_size // int32_size // 2
    node_size = os.path.getsize(NodePath)
    node_num = node_size // int32_size

    data = {}
    data[f"part{rank}"] = {"nodeNUM": node_num,"edgeNUM": edge_num}
    data["path"] = [rank]

    json_string = json.dumps(data, ensure_ascii=False, indent=4)
    json_path = subGSavePath + f"/dist_rank{rank}_{NAME}.json"
    with open(json_path, 'w', encoding='utf-8') as file:
        file.write(json_string)

def partG(args,data,localTrainIds,rank):
    NAME = args.dataset
    nodeNUM = data[NAME]["nodes"]
    RAWPATH = data[NAME]["rawFilePath"]
    subGSavePath = data[NAME]["processedPath"]
    nonNegIdx = torch.nonzero(localTrainIds != -1)[-1]
    localTrainIds = localTrainIds[: nonNegIdx + 1]
    print(f"[{os.getpid()}] (rank = {rank} partition...)")
    
    # find local feature by trainids
    PRgenGByTids(RAWPATH,localTrainIds,nodeNUM,rank,savePath=subGSavePath)

    LABELPATH = RAWPATH + "/labels.bin"
    # trans raw data to local data
    rawData2GNNData(subGSavePath,rank,LABELPATH)
    sliceNUM = 4
    FEATPATH = data[NAME]["rawFilePath"] + "/feat.bin"
    featLen = data[NAME]["featLen"] 
    # dist slice feat get
    genAddFeat(subGSavePath,FEATPATH,rank,nodeNUM,sliceNUM,featLen)
    write_json(NAME,rank,subGSavePath)

def run(args,data):
    NAME = args.dataset
    RAWPATH = data[NAME]["rawFilePath"]
    trainNUM = data[NAME]["trainNUM"]
    TRAINPATH = RAWPATH + "/trainIds.bin"
    
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
    }
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    dist.init_process_group(backend="gloo", timeout=datetime.timedelta(seconds=30))
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    localTrainNUM = (trainNUM + world_size - 1) // world_size
    localTrainIds = torch.zeros(localTrainNUM,dtype=torch.int64)
    if rank == 0:
        trainIds = torch.from_numpy(np.fromfile(TRAINPATH,dtype=np.int64))
        chunkSize = (len(trainIds) + world_size - 1) // world_size
        padded_size = (chunkSize * world_size) - len(trainIds)
        trainIds = torch.cat([trainIds, torch.full((padded_size,), -1, dtype=trainIds.dtype)])#.to("cuda")
        trainIds = list(torch.chunk(trainIds,world_size,dim=0))
        print(trainIds)
        dist.scatter(localTrainIds, src = 0, scatter_list=trainIds)
    else:
        dist.scatter(localTrainIds, src = 0)

    partG(args,data,localTrainIds,rank)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed graph partition")
    parser.add_argument('--dataset', type=str, default='PD', help='dataset name')
    parser.add_argument('--partNUM', type=int, default=8, help='Number of layers')
    args = parser.parse_args()

    JSONPATH = "/Capsule/datasetInfo.json"
    with open(JSONPATH, 'r') as file:
        data = json.load(file)

    run(args,data)


"""
OMP_NUM_THREADS=4 torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr="43.0.0.2" --master_port=1234 dist_part.py
OMP_NUM_THREADS=4 torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr="43.0.0.2" --master_port=1234 dist_part.py


OMP_NUM_THREADS=4 torchrun --nproc_per_node=4 \
         --nnodes=2 \
         --node_rank=1  \ 
         --master_addr="43.0.0.1" \
         --master_port=1234 \
         distDemo.py

torchrun --nproc_per_node=4 \
         --nnodes=2 \
         --node_rank=1\
         --master_addr="192.0.0.1" \
         --master_port=1234\
         trian_multi_node.py
"""