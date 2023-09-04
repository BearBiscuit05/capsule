import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl
import dgl.nn as dglnn
from dgl.data import AsNodePredDataset
from dgl.dataloading import DataLoader, NeighborSampler, MultiLayerFullNeighborSampler
from ogb.nodeproppred import DglNodePropPredDataset
import tqdm
import argparse
import ast
import sklearn.metrics
import numpy as np
import time

class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size,num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, 'mean'))
        for _ in range(num_layers - 2):
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

    def inference(self, g,device, batch_size):
        """Conduct layer-wise inference to get all the node embeddings."""
        feat = g.ndata['feat']
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=['feat'])
        # sampler = NeighborSampler([15],  # fanout for [layer-0, layer-1, layer-2]
        #                     prefetch_node_feats=['feat'],
        #                     prefetch_labels=['label'])
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

def layerwise_infer(device, graph, nid, model, batch_size):
    model.eval()
    with torch.no_grad():
        pred = model.inference(graph, device, batch_size) # pred in buffer_device
        pred = pred[nid]
        label = graph.ndata['label'][nid].to(pred.device)
    return sklearn.metrics.accuracy_score(label.cpu().numpy(), pred.argmax(1).cpu().numpy())

def train(args, device, g, dataset, model,test_idx,data=None):
    # create sampler & dataloader
    if data != None:
        train_idx,val_idx,test_idx = data 
    else:
        train_idx = dataset.train_idx.to(device)
        val_idx = dataset.val_idx.to(device)
        test_idx = dataset.test_idx.to(device)
    # sampler = NeighborSampler([10, 10, 10],  # fanout for [layer-0, layer-1, layer-2]
    #                           prefetch_node_feats=['feat'],
    #                           prefetch_labels=['label'])
    sampler = NeighborSampler(args.fanout,  # fanout for [layer-0, layer-1, layer-2]
                            prefetch_node_feats=['feat'],
                            prefetch_labels=['label'])
    
    use_uva = (args.mode == 'mixed')
    train_dataloader = DataLoader(g, train_idx, sampler, device=device,
                                  batch_size=1024, shuffle=True,
                                  drop_last=False, num_workers=0,
                                  use_uva=use_uva)

    val_dataloader = DataLoader(g, val_idx, sampler, device=device,
                                batch_size=1024, shuffle=True,
                                drop_last=False, num_workers=0,
                                use_uva=use_uva)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    
    epochNum = 200
    epochTime = [0]
    testEpoch = [5,30,50,100,200]
    for epoch in range(1,epochNum+1):
        model.train()
        total_loss = 0
        startTime = time.time()
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            x = blocks[0].srcdata['feat']
            y = blocks[-1].dstdata['label']
            y_hat = model(blocks, x)
            y = y.to(torch.int64)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        eptime = time.time() - startTime
        totTime = epochTime[epoch-1] + eptime
        epochTime.append(totTime)
        acc = evaluate(model, g, val_dataloader)
        print("Epoch {:03d} | Loss {:.4f} | Accuracy {:.4f} | Time {:.6f}"
              .format(epoch, total_loss / (it+1), acc.item(),eptime))
        #save_path = 'model'+str(epoch)+'.pth'
        #torch.save(model.state_dict(), save_path)
        if epoch in testEpoch:
            run_test(args,device,g,dataset,model,test_idx)
    print("Average Time of {:d} Epoches:{:.6f}".format(epochNum,epochTime[epochNum]/epochNum))
    print("Total   Time of {:d} Epoches:{:.6f}".format(epochNum,epochTime[epochNum]))

def load_reddit(self_loop=True):
    from dgl.data import RedditDataset
    data = RedditDataset(self_loop=self_loop,raw_dir='../../../data/dataset/')
    g = data[0]
    g.ndata['feat'] = g.ndata.pop('feat')
    g.ndata['label'] = g.ndata.pop('label')
    train_idx = []
    val_idx = []
    test_idx = []
    for index in range(len(g.ndata['train_mask'])):
        if g.ndata['train_mask'][index] == 1:
            train_idx.append(index)
    for index in range(len(g.ndata['val_mask'])):
        if g.ndata['val_mask'][index] == 1:
            val_idx.append(index)
    for index in range(len(g.ndata['test_mask'])):
        if g.ndata['test_mask'][index] == 1:
            test_idx.append(index)
    return g, data,train_idx,val_idx,test_idx

def run_test(args,device,g,dataset,model,test_idx):
    print('Testing...')
    if args.dataset == 'ogb-products':
        begTime = time.time()
        acc = layerwise_infer(device, g, dataset.test_idx, model, batch_size=4096)
        endTime = time.time()
    elif args.dataset == 'Reddit':
        begTime = time.time()
        acc = layerwise_infer(device, g, test_idx, model, batch_size=4096)
        endTime = time.time()
    elif args.dataset == 'ogb-papers100M':
        model.eval()
        sampler_test = NeighborSampler([100,100],  # fanout for [layer-0, layer-1, layer-2]
                            prefetch_node_feats=['feat'],
                            prefetch_labels=['label'])
        test_dataloader = DataLoader(g, dataset.test_idx, sampler_test, device=device,
                                batch_size=4096, shuffle=True,
                                drop_last=False, num_workers=0,
                                use_uva=True)
        begTime = time.time()
        acc = evaluate(model, g, test_dataloader)
        endTime = time.time()
    print("Test Accuracy {:.4f}".format(acc.item()))
    print('Test Time:',endTime-begTime)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default='mixed', choices=['cpu', 'mixed', 'puregpu'],
                        help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, "
                             "'puregpu' for pure-GPU training.")
    parser.add_argument('--fanout', type=ast.literal_eval, default=[15, 25], help='Fanout value')
    parser.add_argument('--layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--dataset', type=str, default='ogb-products', help='Dataset name')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        args.mode = 'cpu'
    print(f'Training in {args.mode} mode.')
    
    # load and preprocess dataset
    print('Loading data')
    
    
    
    # create GraphSAGE model
    if args.dataset == 'ogb-products':
        dataset = AsNodePredDataset(DglNodePropPredDataset('ogbn-products',root="/home/bear/workspace/singleGNN/data/dataset"))
        g = dataset[0]
        data = None
        test_idx = None
    elif args.dataset == 'Reddit':
        g, dataset,train_idx,val_idx,test_idx= load_reddit()
        data = (train_idx,val_idx,test_idx)
    elif args.dataset == 'ogb-papers100M':
        dataset = AsNodePredDataset(DglNodePropPredDataset('ogbn-papers100M',root="/home/bear/workspace/singleGNN/data/dataset"))
        g = dataset[0]
        data = None
    g = g.to('cuda' if args.mode == 'puregpu' else 'cpu')
    device = torch.device('cpu' if args.mode == 'cpu' else 'cuda')
    
    # create GraphSAGE model

    in_size = g.ndata['feat'].shape[1]
    out_size = dataset.num_classes
    model = SAGE(in_size, 256, out_size,args.layers).to(device)
    # model training
    print('Training...')
    train(args,device,g,dataset,model,test_idx,data=data)
    # model = torch.load("save.pt")
    # model = model.to(device) 
    #model.load_state_dict(torch.load("model_param.pth"))
    # test the model