import os
os.environ['DGLBACKEND'] = 'pytorch'
import dgl
import torch
import numpy as np
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl.nn as dglnn
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import AsNodePredDataset
from dgl.dataloading import DataLoader, NeighborSampler, MultiLayerFullNeighborSampler
import tqdm
import sklearn.metrics
import dgl.nn as dglnn
import time
import sys
import argparse

class GAT(nn.Module):
    def __init__(self,in_size, hid_size, out_size, heads):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GAT
        self.heads = heads
        self.layers.append(dglnn.GATConv(in_size, hid_size, heads[0], feat_drop=0.6, attn_drop=0.6, activation=F.elu,allow_zero_in_degree=True))
        self.layers.append(dglnn.GATConv(hid_size*heads[0], out_size, heads[1], feat_drop=0.6, attn_drop=0.6, activation=None,allow_zero_in_degree=True))
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, g, inputs):
        h = inputs
        for i, layer in enumerate(self.layers):
            h = layer(g[i], h)
            if i == 1:  # last layer 
                h = h.mean(1)
            else:       # other layer(s)
                h = h.flatten(1)
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
                g.num_nodes(), self.hid_size*self.heads[0] if l != len(self.layers) - 1 else self.out_size,
                device=buffer_device, pin_memory=pin_memory)
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x) # len(blocks) = 1
                if l == 1:  # last layer 
                    h = h.mean(1)
                else:       # other layer(s)
                    h = h.flatten(1)
                # by design, our output nodes are contiguous
                y[output_nodes[0]:output_nodes[-1]+1] = h.to(buffer_device)
            feat = y
        return y

def layerwise_infer(device, graph, nid, model, batch_size):
    model.eval()
    with torch.no_grad():
        pred = model.inference(graph, device, batch_size) # pred in buffer_device
        pred = pred[nid]
        label = graph.ndata['label'][nid].to(pred.device)
    return sklearn.metrics.accuracy_score(label.cpu().numpy(), pred.argmax(1).cpu().numpy())

def evaluate(model, graph, dataloader):
    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.no_grad():
            x = blocks[0].srcdata['feat']
            ys.append(blocks[-1].dstdata['label'].cpu().numpy())
            y_hats.append(model(blocks, x).argmax(1).cpu().numpy())
        predictions = np.concatenate(y_hats)
        labels = np.concatenate(ys)
    return sklearn.metrics.accuracy_score(labels, predictions)

def train(args, device, g, train_idx,val_idx, model):
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    use_uva = (args.mode == 'mixed')
    train_dataloader_list = []
    val_dataloader_list = []
    for i in range(4):
        train_dataloader_list.append(
            DataLoader(g[i], train_idx[i], sampler, device=device,
                                  batch_size=4, shuffle=True,
                                  drop_last=False, num_workers=0,
                                  use_uva=use_uva)
        )
        val_dataloader_list.append(
            DataLoader(g[i], val_idx[i], sampler, device=device,
                                batch_size=4, shuffle=True,
                                drop_last=False, num_workers=0,
                                use_uva=use_uva)
        )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    
    for epoch in range(1):
        model.train()
        total_loss = 0
        for i in range(4):
            train_dataloader = train_dataloader_list[i]
            for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
                x = blocks[0].srcdata['feat']
                y = blocks[-1].dstdata['label']
                y_hat = model(blocks, x)
                loss = F.cross_entropy(y_hat, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
                #accuracy = sklearn.metrics.accuracy_score(y.cpu().numpy(), y_hat.argmax(1).detach().cpu().numpy())
            acc = evaluate(model, g, val_dataloader_list[i])
        print("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} "
              .format(epoch, total_loss / (it+1), acc.item()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default='mixed', choices=['cpu', 'mixed', 'puregpu'],
                        help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, "
                             "'puregpu' for pure-GPU training.")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.mode = 'cpu'
    print(f'Training in {args.mode} mode.')
    
    # load and preprocess dataset
    print('Loading data')
    graph_dir = 'data_4/'
    part_config = graph_dir + 'ogb-product.json'
    print('loading partitions')
    
    device = torch.device('cpu' if args.mode == 'cpu' else 'cuda')
    model = GAT(100, 256, 47,heads=[8,1]).to(device)
    g_list = []
    train_list = []
    val_list = []
    for i in range(4):
        subg, node_feat, _, gpb, _, node_type, _ = dgl.distributed.load_partition(part_config, i)
        in_graph = dgl.node_subgraph(subg, subg.ndata['inner_node'].bool())
        in_graph.ndata.clear()
        in_graph.edata.clear()
        in_graph.ndata['feat'] = node_feat['_N/features']
        in_graph.ndata['label'] = node_feat['_N/labels']
        train_mask = node_feat['_N/train_mask']
        train_idx = [index for index, value in enumerate(train_mask) if value == 1]
        train_idx = torch.Tensor(train_idx).to(torch.int64).to(device)
        val_mask = node_feat['_N/val_mask']
        val_idx = [index for index, value in enumerate(val_mask) if value == 1]
        val_idx = torch.Tensor(val_idx).to(torch.int64).to(device)
        subg = subg.to('cuda' if args.mode == 'puregpu' else 'cpu')
        device = torch.device('cpu' if args.mode == 'cpu' else 'cuda')
        g_list.append(in_graph)
        train_list.append(train_idx)
        val_list.append(val_idx)
    print('Training...')
    train(args, device, g_list, train_list , val_list, model)


    dataset = AsNodePredDataset(DglNodePropPredDataset('ogbn-products'))
    g = dataset[0]
    g = g.to('cuda' if args.mode == 'puregpu' else 'cpu')
    device = torch.device('cpu' if args.mode == 'cpu' else 'cuda')
    # test the model
    print('Testing...')
    acc = layerwise_infer(device, g, dataset.test_idx, model, batch_size=1)
    print("Test Accuracy {:.4f}".format(acc.item()))
