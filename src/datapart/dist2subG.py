import argparse
import time
import dgl
import numpy as np
from tools import *
import os
import torch
import torch.distributed as dist
import json
from dist_tools import *

#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
# =============== 1.partition
# WARNING : EDGENUM < 32G Otherwise, it cannot be achieved.
# G_MEM: 16G
MAXEDGE = 900000000    # 
MAXSHUFFLE = 30000000   # 
#################

batch = None

def gen_edge_batch(src,dst):
    global batch
    if batch is None:
        batch_size = len(src) // MAXEDGE + 1
        src_batches = torch.chunk(src, batch_size, dim=0)
        dst_batches = torch.chunk(dst, batch_size, dim=0)
        batch = [src_batches, dst_batches]
        return batch
    else:
        return batch
def sum_node_degree(src,dst,node_num):
    # Streaming edge data for compute node degree 
    batch = gen_edge_batch(src,dst)
    inNodeTable = torch.zeros(node_num,dtype=torch.int32,device="cuda")
    outNodeTable = torch.zeros(node_num,dtype=torch.int32,device="cuda")
    
    for src_batch,dst_batch in zip(*batch):
        src_batch,dst_batch = src_batch.cuda(),dst_batch.cuda()
        inNodeTable,outNodeTable = dgl.sumDegree(inNodeTable,outNodeTable,src_batch,dst_batch)
    src_batch,dst_batch = None,None
    outNodeTable = outNodeTable.cpu() 
    del outNodeTable
    
    # innodeTable still in GPU for next use
    return inNodeTable

################### [1] PRgenG
def PRgenG(rank,RAWPATH,trainIds,nodeNUM,per_rank_part,savePath=None):
    GRAPHPATH = RAWPATH + "/graph.bin"
    graph = torch.from_numpy(np.fromfile(GRAPHPATH,dtype=np.int32))
    src,dst = graph[::2],graph[1::2]
    edgeNUM = len(src)
    
    for i in range(per_rank_part):
        PATH = savePath + f"/part{i}" 
        checkFilePath(PATH)

    inNodeTable = sum_node_degree(src,dst,nodeNUM)

    nodeInfo = torch.zeros(nodeNUM,dtype=torch.int32)
    nodeValue = torch.zeros(nodeNUM,dtype=torch.int32)
    # value setting
    nodeValue[trainIds] = 100000

    shuffled_indices = torch.randperm(trainIds.size(0))
    trainIds = trainIds[shuffled_indices]
    trainBatch = torch.chunk(trainIds, per_rank_part, dim=0)
    for index,ids in enumerate(trainBatch):
        info = 1 << index
        nodeInfo[ids] = info
        PATH = savePath + f"/part{index}" 
        TrainPath = PATH + f"/raw_trainIds.bin"
        saveBin(ids,TrainPath)
    # ====

    batch = gen_edge_batch(src,dst)
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

    for bit_position in range(per_rank_part):
        # GPU : nodeIndex,outIndex
        nodeIndex = (nodeInfo & (1 << bit_position)) != 0
        outIndex  =  (outlayer & (1 << bit_position)) != 0  # Indicates whether it is a three-hop point
        nid = torch.nonzero(nodeIndex).reshape(-1).to(torch.int32).cpu()
        PATH = savePath + f"/part{bit_position}"
        checkFilePath(PATH)
        DataPath = PATH + f"/raw_G.bin"
        NodePath = PATH + f"/raw_nodes.bin"
        PRvaluePath = PATH + f"/sortIds.bin"
        saveBin(nid,NodePath)
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
def rawData2GNNData(rank,RAWDATAPATH,per_rank_part,LABELPATH):
    labels = np.fromfile(LABELPATH,dtype=np.int64)  
    for part_id in range(per_rank_part):
        part_process(part_id,RAWDATAPATH,labels)
        emptyCache()

def part_process(part_id,RAWDATAPATH,labels):
    startTime = time.time()
    prefix_path = RAWDATAPATH + f"/part{part_id}" 
    rawDataPath = prefix_path + f"/raw_G.bin"
    rawTrainPath = prefix_path + f"/raw_trainIds.bin"
    rawNodePath = prefix_path + f"/raw_nodes.bin"
    PRvaluePath = prefix_path + f"/sortIds.bin"
    SubTrainIdPath = prefix_path + "/trainIds.bin"
    SubIndptrPath = prefix_path + "/indptr.bin"
    SubIndicesPath = prefix_path + "/indices.bin"
    SubLabelPath = prefix_path + "/labels.bin"
    checkFilePath(prefix_path)

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

    dataInfo[f"part{part_id}"] = {'nodeNUM': len(node),'edgeNUM':len(data) // 2}
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

def genAddFeat(beginId,addIdx,SAVEPATH,FEATPATH,partNUM,nodeNUM,sliceNUM,featLen):
    # addIdx now in CUDA
    emptyCache()
    slice = nodeNUM // sliceNUM + 1
    boundList = [0]
    start = slice
    for i in range(sliceNUM):
        boundList.append(start)
        start += slice
    boundList[-1] = nodeNUM

    file = SAVEPATH + f"/part{beginId}/raw_nodes.bin"
    ids = torch.as_tensor(np.fromfile(file,dtype=np.int32),device="cuda")
    addIdx.append(ids)  # Increases all indexes of the original loaded subgraph

    for i in range(partNUM+1):
        addIdx[i] = sliceIds(addIdx[i],boundList)

    for sliceIndex in range(sliceNUM):
        beginIdx = boundList[sliceIndex]
        endIdx = boundList[sliceIndex+1]
        sliceFeat = featSlice(FEATPATH,beginIdx,endIdx,featLen).cuda()
        for index in range(partNUM + 1):
            if index == partNUM:
                fileName = SAVEPATH + f"/part{beginId}/feat.bin"
            else:
                fileName = SAVEPATH + f"/part{index}/addfeat.bin"
            SubIdsList = addIdx[index][sliceIndex]
            t_SubIdsList = SubIdsList - beginIdx
            addFeat = sliceFeat[t_SubIdsList.to(torch.int64)]   # t_SubIdsList is in CUDA
            addFeat = addFeat.cpu()
            saveBin(addFeat,fileName,addSave=sliceIndex)


# =============== 4.addFeat


def dfs(part_num, diffMatrix, cur, res, cur_sum, res_sum):
    if len(cur) == part_num:
        if res_sum[0] == -1 or cur_sum < res_sum[0]:
            res[0] = cur[:]
            res_sum[0] = cur_sum
        return
    
    for i in range(part_num):
        if i in cur or (res_sum[0] != -1 and len(cur) > 0 and cur_sum + diffMatrix[cur[-1]][i] > res_sum[0]):
            continue
        
        if len(cur) != 0:
            cur_sum += diffMatrix[cur[-1]][i]
        
        cur.append(i)
        dfs(part_num, diffMatrix, cur, res, cur_sum, res_sum)
        cur.pop()
        
        if len(cur) != 0:
            cur_sum -= diffMatrix[cur[-2]][i]

def cal_min_path(diffMatrix, nodesList, part_num, base_path):
    base_path += '/part'
    start = time.time()
    maxNodeNum = 0
    for i in range(part_num):
        path = base_path + str(i) + '/raw_nodes.bin'
        nodes = torch.as_tensor(np.fromfile(path, dtype=np.int32)).cuda()
        maxNodeNum = max(maxNodeNum, nodes.shape[0])
        nodesList.append(nodes)
    
    res1 = torch.zeros(maxNodeNum, dtype=torch.int32,device="cuda")
    res2 = torch.zeros(maxNodeNum, dtype=torch.int32,device="cuda")
    print(f"load all nodes {time.time() - start:.4f}s")
    for i in range(part_num):
        for j in range(i + 1,part_num):
            node1 = nodesList[i]
            node2 = nodesList[j]
            res1.fill_(0)
            res2.fill_(0)
            dgl.findSameNode(node1, node2, res1, res2)
            sameNum = torch.sum(res1).item()
            diffMatrix[i][j] = node2.shape[0] - sameNum # The additional loading required for j relative to i
            diffMatrix[j][i] = node1.shape[0] - sameNum


    start = time.time()
    res = [[]]
    res_sum = [-1]
    dfs(part_num, diffMatrix, [], res, 0, res_sum)
    print("dfs time: {}".format(time.time() - start))
    return maxNodeNum, res

def genFeatIdx(part_num, base_path, nodeList, part_seq, featLen, maxNodeNum):
    res1 = torch.zeros(maxNodeNum, dtype=torch.int32,device="cuda")
    res2 = torch.zeros(maxNodeNum, dtype=torch.int32,device="cuda")
    base_path += '/part'
    addIndex = [[] for _ in range(part_num)]
    for i in range(0, part_num):
        cur_part = part_seq[i]
        next_part = part_seq[(i+1) % part_num]
        curNode = nodeList[cur_part].cuda()
        nextNode = nodeList[next_part].cuda()
        curLen = curNode.shape[0]
        nextLen = nextNode.shape[0]
        
        res1.fill_(0)
        res2.fill_(0)
        dgl.findSameNode(curNode, nextNode, res1, res2)
        same_num = torch.sum(res1).item()
        
        # index
        maxlen = max(curLen,nextLen)
        res1_zero = torch.nonzero(res1[:maxlen] == 0).reshape(-1).to(torch.int32)
        res2_zero = torch.nonzero(res2[:maxlen] == 0).reshape(-1).to(torch.int32)
        res1_one = torch.nonzero(res1[:maxlen] == 1).reshape(-1).to(torch.int32)
        res2_one = torch.nonzero(res2[:maxlen] == 1).reshape(-1).to(torch.int32)

        if (nextLen > same_num):
            if(curLen < nextLen or curLen == nextLen):
                replaceIdx = res2_zero.cuda()
            elif(curLen > nextLen):
                replaceIdx = res2_zero[:nextLen - same_num].cuda()  # If the feat index is not clipped, the feat index is out of bounds
        else:
            replaceIdx = torch.Tensor([]).to(torch.int32)
            

        nextPath = base_path + str(next_part)
        sameNodeInfoPath = nextPath + '/sameNodeInfo.bin'
        diffNodeInfoPath = nextPath + '/diffNodeInfo.bin'
        if res1_one.shape[0] != 0:
            sameNode = torch.cat((res1_one, res2_one), dim = 0)
        else:
            sameNode = torch.Tensor([]).to(torch.int32)
        
        if res1_zero.shape[0] != 0:
            diffNode = torch.cat((res1_zero, res2_zero), dim = 0)
        else:
            diffNode = torch.Tensor([]).to(torch.int32)
        saveBin(sameNode.cpu(), sameNodeInfoPath)
        saveBin(diffNode.cpu(), diffNodeInfoPath)
        sameNode, diffNode = None, None

        # Cache the index that needs to be reloaded
        addIndex[next_part] = nodeList[next_part][replaceIdx.to(torch.int64)]
    return addIndex

def writeJson(path):
    with open(path, "w") as json_file:
        json.dump(dataInfo, json_file,indent=4)



def partG(args,data,localTrainIds,rank):
    # TODO: to cuda
    nonNegIdx = torch.nonzero(localTrainIds != -1)[-1]
    localTrainIds = localTrainIds[: nonNegIdx + 1]
    print(f"[{os.getpid()}] (rank = {rank} partition...)")

    nodeNUM = data[args.dataset]["nodes"]
    RAWPATH = data[args.dataset]["rawFilePath"]
    subGSavePath = data[args.dataset]["processedPath"]+ f"/rank{rank}"
    LABELPATH = RAWPATH + "/labels.bin"
    FEATPATH = data[args.dataset]["rawFilePath"] + "/feat.bin"
    featLen = data[args.dataset]["featLen"]

    # find local feature by trainids
    per_rank_part = 1
    PRgenG(rank,RAWPATH,localTrainIds,nodeNUM,per_rank_part,savePath=subGSavePath)

    # trans raw data to local data
    rawData2GNNData(rank,subGSavePath,per_rank_part,LABELPATH)

    if per_rank_part > 1:
        diffMatrix = [[0 for _ in range(per_rank_part)] for _ in range(per_rank_part)]
        nodeList = []
        maxNodeNum,minPath = cal_min_path(diffMatrix , nodeList, per_rank_part, subGSavePath)
    else:
        minPath = [0]
        path = subGSavePath + "/part0/raw_nodes.bin"
        single_node = torch.as_tensor(np.fromfile(path, dtype=np.int32)).cuda()
        maxNodeNum = single_node.shape[0]
        nodeList = [single_node]
        # dataInfo['path'] = minPath
        # writeJson(SAVEPATH+f"/{NAME}.json")

    sliceNUM = 4
    addIdx = genFeatIdx(per_rank_part, subGSavePath, nodeList, minPath, featLen, maxNodeNum)
    genAddFeat(minPath[0],addIdx,subGSavePath,FEATPATH,per_rank_part,nodeNUM,sliceNUM,featLen)

    
    dataInfo['path'] = minPath
    writeJson(subGSavePath+f"/{args.dataset}.json")




# ======= dist =======
def run(args, data):
    NAME = args.dataset
    train_num = data[NAME]["trainNUM"]
    TRAINPATH = data[NAME]["rawFilePath"] + "/trainIds.bin"
    
    init_process_group()
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    localTrainIds = get_local_train_ids(world_size, train_num)

    if rank == 0:
        trainIds_chunks = prepare_train_ids(TRAINPATH, world_size)
        scatter_train_ids(rank, localTrainIds, trainIds_chunks)
    else:
        scatter_train_ids(rank, localTrainIds)

    partG(args, data, localTrainIds, rank)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed graph partition")
    parser.add_argument('--dataset', type=str, default='PD', help='dataset name')
    parser.add_argument('--per_part', type=int, default=1, help='Number of layers')
    args = parser.parse_args()

    JSONPATH = "/Capsule/datasetInfo.json"
    with open(JSONPATH, 'r') as file:
        data = json.load(file)

    run(args,data)


"""
OMP_NUM_THREADS=4 torchrun --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr="43.0.0.2" --master_port=1234 dist2subG.py
OMP_NUM_THREADS=4 torchrun --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr="43.0.0.2" --master_port=1234 dist2subG.py
"""