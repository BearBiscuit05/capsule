import dgl
import torch
import numpy as np

RAWPATH = "capsule/data/raw/wb2001"
featlen = 100
Ratio = 0.01    # train id ratio

graph = torch.as_tensor(np.fromfile(RAWPATH+"/graph.bin",dtype=np.int32))
maxId = torch.max(graph) + 1    # from 0 - (maxId-1)

feat = torch.zeros((maxId,featlen),dtype=torch.float32)
labels = torch.ones(maxId,dtype=torch.int64)
trainIds = torch.randperm(maxId)[:int(maxId*Ratio)].to(torch.int64)

print("gen feat/label/trainIds success...")

feat.numpy().tofile(RAWPATH+"/feat.bin")
labels.numpy().tofile(RAWPATH+"/labels.bin")
trainIds.numpy().tofile(RAWPATH+"/trainIds.bin")

print("custum dataset save success...")