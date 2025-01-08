import argparse
import os
import sys
from time import sleep
from urllib.parse import urlparse

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import datetime

import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl
import dgl.nn as dglnn
from dgl.data import AsNodePredDataset
from dgl.dataloading import DataLoader, NeighborSampler, MultiLayerFullNeighborSampler

from ogb.nodeproppred import DglNodePropPredDataset
import tqdm
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import os
import sys

curDir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(curDir+"/../"+"load")
from distLoader import DistDataset

class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, 'mean'))
        self.layers.append(dglnn.SAGEConv(hid_size, hid_size, 'mean'))
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, 'mean'))
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, g, device, batch_size):
        """Conduct layer-wise inference to get all the node embeddings."""
        feat = g.ndata['feat']
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=['feat'])
        dataloader = DataLoader(
                g, torch.arange(g.num_nodes()).to(g.device), sampler, device=device,
                batch_size=batch_size, shuffle=False, drop_last=False,
                num_workers=0)
        buffer_device = torch.device('cpu')
        pin_memory = (buffer_device != device)

        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(), self.hid_size if l != len(self.layers) - 1 else self.out_size,
                device=buffer_device, pin_memory=pin_memory)
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x) # len(blocks) = 1
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                # by design, our output nodes are contiguous
                y[output_nodes[0]:output_nodes[-1]+1] = h.to(buffer_device)
            feat = y
        return y
    
def evaluate(model, graph, dataloader, num_classes):
    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.no_grad():
            x = blocks[0].srcdata["feat"]
            ys.append(blocks[-1].dstdata["label"])
            y_hats.append(model(blocks, x))
    return MF.accuracy(
        torch.cat(y_hats),
        torch.cat(ys),
        task="multiclass",
        num_classes=num_classes,
    )

def layerwise_infer(device, graph, nid, model, num_classes, batch_size):
    model.eval()
    with torch.no_grad():
        pred = model.inference(
            graph, device, batch_size
        )  # pred in buffer_device
        pred = pred[nid]
        label = graph.ndata["label"][nid].to(pred.device)
        return MF.accuracy(
            pred, label, task="multiclass", num_classes=num_classes
        )

def Testing(model,name,data,device="cuda",tid=None,num_classes=0):
    if name == "PD":
        g = data[0]
        acc = layerwise_infer(device, g, data.test_idx, model, num_classes, batch_size=4096)
    elif name == "RD":
        acc = layerwise_infer(device, data, tid, model, num_classes, batch_size=4096)  
    print("Test Accuracy {:.4f}".format(acc.item()))


def collate_fn(data):
    return data[0]

def train():    
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    PATH = "/sgnn/data/distPartition/PD/part"
    indices = np.fromfile(PATH + f"{rank}/indices.bin",dtype=np.int32)
    indptr = np.fromfile(PATH + f"{rank}/indptr.bin",dtype=np.int32)
    indptr = torch.tensor(indptr).to(torch.int64)
    indices = torch.tensor(indices).to(torch.int64)
    feat = np.fromfile(PATH + f"{rank}/feat.bin",dtype=np.float32).reshape(-1,100)
    device = torch.device('cuda')
    trainIds = np.fromfile(PATH + f"{rank}/trainIds.bin",dtype=np.int64)
    trainIds = torch.tensor(trainIds).to(torch.int64).to(device)
    labels = np.fromfile(PATH + f"{rank}/labels.bin",dtype=np.int64)
    
    g = dgl.graph(('csc', (indptr, indices, []))).to(device)        
    feat = torch.tensor(feat).to(device)
    g.ndata['feat'] = feat
    labels = torch.tensor(labels).to(torch.int64).to(device)
    g.ndata['label'] = labels
    print(f"[{os.getpid()}] (rank = {rank}) training,with {indices.shape}...")
    model = SAGE(100, 256, 47).to(device)
    model = DDP(model,[local_rank])


    # TODO : 替换
    sampler = NeighborSampler([10, 10, 10],  # fanout for [layer-0, layer-1, layer-2]
                              prefetch_node_feats=['feat'],
                              prefetch_labels=['label'])
    train_dataloader = DataLoader(g, trainIds, sampler, device=device,
                                  batch_size=512, shuffle=True,
                                  drop_last=False, num_workers=0,
                                  use_uva=False)
    
    #train_dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=dataset.batchsize, collate_fn=collate_fn)#,pin_memory=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    for epoch in range(20):
        model.train()
        total_loss = 0
        startTime = time.time()
        with model.join():
            for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
                x = blocks[0].srcdata['feat']
                y = blocks[-1].dstdata['label']
                y_hat = model(blocks, x)
                loss = F.cross_entropy(y_hat, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
            
            # for it,(graph,feat,label,number) in enumerate(train_dataloader):
            #     feat = feat.cuda()
            #     y_hat = model(graph, feat)
            #     label = label.to(torch.int64)
            #     loss = F.cross_entropy(y_hat[:number], label[:number].to(y_hat.device))
            #     opt.zero_grad()
            #     loss.backward()
            #     opt.step()
            #     total_loss += loss.item()
        
        
        
        acc = torch.Tensor([0.00])
        print("Epoch {:05d} | Time {:.3f}s | Loss {:.4f} | Accuracy {:.4f} "
              .format(epoch,time.time()-startTime ,total_loss / ((it+1) * 1), acc.item()))
    if rank == 0:
        torch.save(model.module.state_dict(), "./model/model.pth")
        evamodel = SAGE(100, 256, 47).to(device)
        evamodel.load_state_dict(torch.load('./model/model.pth'))
        dataset = AsNodePredDataset(DglNodePropPredDataset('ogbn-products',root="/sgnn/data/partition/"))
        Testing(evamodel,"PD",dataset,num_classes=47)


def run():
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
    }
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    dist.init_process_group(backend="gloo", timeout=datetime.timedelta(seconds=30))
    train()
    dist.destroy_process_group()


if __name__ == "__main__":
    run()