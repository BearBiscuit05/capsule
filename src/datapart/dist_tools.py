"""
a module for dist partitioning
"""

import numpy as np
import os
import torch
import torch.distributed as dist
import datetime

def init_process_group(backend="gloo", timeout_seconds=30):
    """Initialize the distributed process group."""
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
    }
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    dist.init_process_group(
        backend=backend, 
        timeout=datetime.timedelta(seconds=timeout_seconds)
    )


def prepare_train_ids(train_path, world_size):
    """Read, shuffle and chunk train IDs into equal parts for each worker."""
    trainIds = torch.from_numpy(np.fromfile(train_path, dtype=np.int64))
    perm = torch.randperm(len(trainIds))
    trainIds = trainIds[perm]

    chunkSize = (len(trainIds) + world_size - 1) // world_size
    padded_size = (chunkSize * world_size) - len(trainIds)

    if padded_size > 0:
        padding = torch.full((padded_size,), -1, dtype=trainIds.dtype)
        trainIds = torch.cat([trainIds, padding])
    
    return list(torch.chunk(trainIds, world_size, dim=0))


def scatter_train_ids(rank, localTrainIds, trainIds_chunks=None, src=0):
    """Scatter train IDs to all workers."""
    if rank == src and trainIds_chunks is not None:
        dist.scatter(localTrainIds, src=src, scatter_list=trainIds_chunks)
    else:
        dist.scatter(localTrainIds, src=src)


def get_local_train_ids(world_size, trainNUM):
    """Calculate the number of training IDs for each worker."""
    local_train_num = (trainNUM + world_size - 1) // world_size
    return torch.zeros(local_train_num, dtype=torch.int64)

