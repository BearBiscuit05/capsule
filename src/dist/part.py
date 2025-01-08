import numpy as np
import dgl
import torch
import os
import copy
import gc
import time
import json
import sys
import argparse

curDir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(curDir+"/../"+"tools")
from tools import *
# from subCluster import *
# =============== 1.partition
# WARNING : EDGENUM < 32G Otherwise, it cannot be achieved.
# G_MEM: 16G
MAXEDGE = 900000000    # 
MAXSHUFFLE = 30000000   # 
#################

## pagerank+label Traversal gets the base subgraph
def PRgenGByTids(RAWPATH,trainIds,nodeNUM,Rank,savePath=None):
    GRAPHPATH = RAWPATH + "/graph.bin"
    PATH = savePath + f"/part{Rank}" 
    checkFilePath(PATH)
    
    graph = torch.from_numpy(np.fromfile(GRAPHPATH,dtype=np.int32))
    src,dst = graph[::2],graph[1::2]
    edgeNUM = len(src)
    template_array = torch.zeros(nodeNUM,dtype=torch.int32)

    # Streaming edge data
    batch_size = len(src) // MAXEDGE + 1
    src_batches = torch.chunk(src, batch_size, dim=0)
    dst_batches = torch.chunk(dst, batch_size, dim=0)
    batch = [src_batches, dst_batches]

    inNodeTable, outNodeTable = template_array.clone().cuda(), template_array.clone().cuda()
    for src_batch,dst_batch in zip(*batch):
        src_batch,dst_batch = src_batch.cuda(),dst_batch.cuda()
        inNodeTable,outNodeTable = dgl.sumDegree(inNodeTable,outNodeTable,src_batch,dst_batch)
    src_batch,dst_batch = None,None
    outNodeTable = outNodeTable.cpu() # innodeTable still in GPU for next use

    nodeValue = template_array.clone()
    nodeInfo = template_array.clone()
    # value setting
    nodeValue[trainIds] = 100000

    # random method
    info = 1 << 0
    nodeInfo[trainIds] = info
    PATH = savePath + f"/part{Rank}" 
    TrainPath = PATH + f"/raw_trainIds.bin"
    saveBin(trainIds,TrainPath)
    # ====

    nodeLayerInfo = []
    for _ in range(3):
        offset = 0
        acc_nodeValue = torch.zeros_like(nodeValue,dtype=torch.int32)
        acc_nodeInfo = torch.zeros_like(nodeInfo,dtype=torch.int32)
        for src_batch,dst_batch in zip(*batch):  
            tmp_nodeValue,tmp_nodeInfo = nodeValue.clone().cuda(),nodeInfo.clone().cuda() 
            src_batch,dst_batch = src_batch.cuda(), dst_batch.cuda()  
            dgl.per_pagerank(dst_batch,src_batch,inNodeTable,tmp_nodeValue,tmp_nodeInfo)
            tmp_nodeValue, tmp_nodeInfo = tmp_nodeValue.cpu(),tmp_nodeInfo.cpu()
            acc_nodeValue += tmp_nodeValue - nodeValue
            acc_nodeInfo = acc_nodeInfo | tmp_nodeInfo
            offset += len(src_batch)
        nodeValue = nodeValue + acc_nodeValue
        nodeInfo = acc_nodeInfo
        tmp_nodeValue,tmp_nodeInfo=None,None
        nodeLayerInfo.append(nodeInfo.clone())
    src_batch,dst_batch,inNodeTable = None,None,None
    outlayer = torch.bitwise_xor(nodeLayerInfo[-1], nodeLayerInfo[-2]) # The outermost point will not have a connecting edge
    nodeLayerInfo = None
    emptyCache()
    nodeInfo = nodeInfo.cuda()
    outlayer = outlayer.cuda()
    bit_position = 0
    nodeIndex = (nodeInfo & (1 << bit_position)) != 0
    outIndex  =  (outlayer & (1 << bit_position)) != 0  # Indicates whether it is a three-hop point
    nid = torch.nonzero(nodeIndex).reshape(-1).to(torch.int32).cpu()
    PATH = savePath + f"/part{Rank}" 
    DataPath = PATH + f"/raw_G.bin"
    NodePath = PATH + f"/raw_nodes.bin"
    PRvaluePath = PATH + f"/sortIds.bin"
    selfLoop = np.repeat(trainIds.to(torch.int32), 2)
    saveBin(nid,NodePath)
    saveBin(selfLoop,DataPath)
    graph = graph.reshape(-1,2)
    sliceNUM = (edgeNUM-1) // (MAXEDGE//2) + 1
    offsetSize = (edgeNUM-1) // sliceNUM + 1
    offset = 0
    start = time.time()
    for i in range(sliceNUM):
        sliceLen = min((i+1)*offsetSize,edgeNUM)
        g_gpu = graph[offset:sliceLen]                  # part of graph
        g_gpu = g_gpu.cuda()
        gsrc,gdst = g_gpu[:,0],g_gpu[:,1]
        gsrcMask = nodeIndex[gsrc.to(torch.int64)]
        gdstMask = nodeIndex[gdst.to(torch.int64)]
        idx_gpu = torch.bitwise_and(gsrcMask, gdstMask) # This time also includes a triple jump side
        IsoutNode = outIndex[gdst.to(torch.int64)]
        idx_gpu = torch.bitwise_and(idx_gpu, ~IsoutNode) # The three-hop edge has been deleted
        subEdge = g_gpu[idx_gpu].cpu()
        saveBin(subEdge,DataPath,addSave=True)
        offset = sliceLen                       
    print(f"time :{time.time()-start:.3f}s")    
    partValue = nodeValue[nodeIndex]  
    _ , sort_indice = torch.sort(partValue,dim=0,descending=True)
    sort_nodeid = nid[sort_indice]
    saveBin(sort_nodeid,PRvaluePath)

# =============== 2.graphToSub    
def nodeShuffle(raw_node,raw_graph):
    srcs, dsts = raw_graph[::2], raw_graph[1::2]
    raw_node = convert_to_tensor(raw_node, dtype=torch.int32).cuda()
    srcs_tensor = convert_to_tensor(srcs, dtype=torch.int32)
    dsts_tensor = convert_to_tensor(dsts, dtype=torch.int32)
    uniTable = torch.ones(len(raw_node),dtype=torch.int32,device="cuda")
    batch_size = len(srcs) // (MAXEDGE//2) + 1
    src_batches = list(torch.chunk(srcs_tensor, batch_size, dim=0))
    dst_batches = list(torch.chunk(dsts_tensor, batch_size, dim=0))
    batch = [src_batches, dst_batches]
    src_emp,dst_emp = raw_node[:1].clone(), raw_node[:1].clone()    # padding , no use
    srcShuffled,dstShuffled,uniTable = dgl.mapByNodeSet(raw_node,uniTable,src_emp,dst_emp,rhsNeed=False,include_rhs_in_lhs=False)
    raw_node = raw_node.cpu()
    remap = None
    for index,(src_batch,dst_batch) in enumerate(zip(*batch)):
        srcShuffled,dstShuffled,remap = remapEdgeId(uniTable,src_batch,dst_batch,remap=remap,device=torch.device('cuda:0'))
        src_batches[index] = srcShuffled
        dst_batches[index] = dstShuffled 
    srcShuffled,dstShuffled=None,None
    srcs_tensor = torch.cat(src_batches).cpu()
    dsts_tensor = torch.cat(dst_batches).cpu()
    uniTable = uniTable.cpu()
    return srcs_tensor,dsts_tensor,uniTable

def trainIdxSubG(subGNode,trainSet):
    trainSet = torch.as_tensor(trainSet).to(torch.int32)
    Lid = torch.zeros_like(trainSet).to(torch.int32).cuda()
    dgl.mapLocalId(subGNode.cuda(),trainSet.cuda(),Lid)
    Lid = Lid.cpu().to(torch.int64)
    return Lid

dataInfo = {}
def rawData2GNNData(RAWDATAPATH,rank,LABELPATH):
    labels = np.fromfile(LABELPATH,dtype=np.int64)  
    partProcess(rank,RAWDATAPATH,labels)

def partProcess(rank,RAWDATAPATH,labels):
    startTime = time.time()
    PATH = RAWDATAPATH + f"/part{rank}" 
    rawDataPath = PATH + f"/raw_G.bin"
    rawTrainPath = PATH + f"/raw_trainIds.bin"
    rawNodePath = PATH + f"/raw_nodes.bin"
    PRvaluePath = PATH + f"/sortIds.bin"
    SubTrainIdPath = PATH + "/trainIds.bin"
    SubIndptrPath = PATH + "/indptr.bin"
    SubIndicesPath = PATH + "/indices.bin"
    SubLabelPath = PATH + "/labels.bin"
    checkFilePath(PATH)
    coostartTime = time.time()
    data = np.fromfile(rawDataPath,dtype=np.int32)
    node = np.fromfile(rawNodePath,dtype=np.int32)
    trainidx = np.fromfile(rawTrainPath,dtype=np.int64)  
    print(f"loading data time : {time.time()-coostartTime:.4f}s")
    
    coostartTime = time.time()
    remappedSrc,remappedDst,uniNode = nodeShuffle(node,data)
    subLabel = labels[uniNode.to(torch.int64)]
    indptr, indices = cooTocsc(remappedSrc,remappedDst,sliceNUM=(len(data) // (MAXEDGE//2))) 
    print(f"coo data time : {time.time()-coostartTime:.4f}s")

    coostartTime = time.time()
    trainidx = trainIdxSubG(uniNode,trainidx)
    saveBin(subLabel,SubLabelPath)
    saveBin(trainidx,SubTrainIdPath)
    saveBin(indptr,SubIndptrPath)
    saveBin(indices,SubIndicesPath)
    print(f"save time : {time.time()-coostartTime:.4f}s")
    
    pridx = torch.as_tensor(np.fromfile(PRvaluePath,dtype=np.int32))
    remappedSrc,_,_ = remapEdgeId(uniNode,pridx,None,device=torch.device('cuda:0'))
    saveBin(remappedSrc,PRvaluePath)

    dataInfo[f"part{rank}"] = {'nodeNUM': len(node),'edgeNUM':len(data) // 2}
    print(f"map data time : {time.time()-startTime:.4f}s")
    print("-"*20)



# =============== 3.featTrans
def featSlice(FEATPATH,beginIndex,endIndex,featLen):
    blockByte = 4 # float32 4byte
    offset = (featLen * beginIndex) * blockByte
    subFeat = torch.as_tensor(np.fromfile(FEATPATH, dtype=np.float32, count=(endIndex - beginIndex) * featLen, offset=offset))
    return subFeat.reshape(-1,featLen)

def sliceIds(Ids,sliceTable):
    # Cut Ids into the range specified by sliceTable
    # Ids can only be the sorted result
    beginIndex = 0
    ans = []
    for tar in sliceTable[1:]:
        position = torch.searchsorted(Ids, tar)
        slice = Ids[beginIndex:position]
        ans.append(slice)
        beginIndex = position
    return ans

def genSubGFeat(SAVEPATH,FEATPATH,partNUM,nodeNUM,sliceNUM,featLen):
    # get slices
    emptyCache()
    slice = nodeNUM // sliceNUM + 1
    boundList = [0]
    start = slice
    for i in range(sliceNUM):
        boundList.append(start)
        start += slice
    boundList[-1] = nodeNUM

    idsSliceList = [[] for i in range(partNUM)]
    for i in range(partNUM):
        file = SAVEPATH + f"/part{i}/raw_nodes.bin"
        ids = torch.as_tensor(np.fromfile(file,dtype=np.int32))
        idsSliceList[i] = sliceIds(ids,boundList)
    
    for sliceIndex in range(sliceNUM):
        beginIdx = boundList[sliceIndex]
        endIdx = boundList[sliceIndex+1]
        sliceFeat = featSlice(FEATPATH,beginIdx,endIdx,featLen).cuda()
        for index in range(partNUM):
            fileName = SAVEPATH + f"/part{index}/feat.bin"
            SubIdsList = idsSliceList[index][sliceIndex]
            t_SubIdsList = SubIdsList - beginIdx
            subFeat = sliceFeat[t_SubIdsList.to(torch.int64).cuda()]
            subFeat = subFeat.cpu()
            saveBin(subFeat,fileName,addSave=sliceIndex)

def genAddFeat(SAVEPATH,FEATPATH,rank,nodeNUM,sliceNUM,featLen):
    nodesList = []
    #for i in range(partNUM):
    path = SAVEPATH + "/part" + str(rank) + '/raw_nodes.bin'
    nodes = torch.as_tensor(np.fromfile(path, dtype=np.int32)).cuda()
    #nodesList.append(nodes)
    
    slice = nodeNUM // sliceNUM + 1
    boundList = [0]
    start = slice
    for i in range(sliceNUM):
        boundList.append(start)
        start += slice
    boundList[-1] = nodeNUM

    # for i in range(partNUM):
    nodes = sliceIds(nodes,boundList)

    for sliceIndex in range(sliceNUM):
        beginIdx = boundList[sliceIndex]
        endIdx = boundList[sliceIndex+1]
        sliceFeat = featSlice(FEATPATH,beginIdx,endIdx,featLen).cuda()
        #for index in range(partNUM):
        fileName = SAVEPATH + f"/part{rank}/feat.bin"
        SubIdsList = nodes[sliceIndex]
        t_SubIdsList = SubIdsList - beginIdx
        addFeat = sliceFeat[t_SubIdsList.to(torch.int64)]   # t_SubIdsList is in CUDA
        addFeat = addFeat.cpu()
        saveBin(addFeat,fileName,addSave=sliceIndex)


def writeJson(path):
    with open(path, "w") as json_file:
        json.dump(dataInfo, json_file,indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='PD', help='dataset name')
    parser.add_argument('--partNUM', type=int, default=8, help='Number of layers')
    args = parser.parse_args()

    JSONPATH = "/sgnn/datasetInfo.json"
    partitionNUM = args.partNUM
    sliceNUM = 8
    with open(JSONPATH, 'r') as file:
        data = json.load(file)

    NAME = args.dataset
    GRAPHPATH = data[NAME]["rawFilePath"]
    maxID = data[NAME]["nodes"]
    subGSavePath = data[NAME]["processedPath"]
    
    startTime = time.time()
    PRgenGByTids(GRAPHPATH,maxID,partitionNUM,savePath=subGSavePath)
    print(f"partition all cost:{time.time()-startTime:.3f}s")

    RAWDATAPATH = data[NAME]["processedPath"]
    FEATPATH = data[NAME]["rawFilePath"] + "/feat.bin"
    LABELPATH = data[NAME]["rawFilePath"] + "/labels.bin"
    SAVEPATH = data[NAME]["processedPath"]
    nodeNUM = data[NAME]["nodes"]
    featLen = data[NAME]["featLen"] 
    MERGETIME = time.time()
    rawData2GNNData(RAWDATAPATH,partitionNUM,LABELPATH)
    print(f"trans graph cost time{time.time() - MERGETIME:.3f}s ...")
    
    genAddFeat(SAVEPATH,FEATPATH,partitionNUM,nodeNUM,sliceNUM,featLen)