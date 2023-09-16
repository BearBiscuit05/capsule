#include "readGraph.h"

ReadEngine::ReadEngine() {}

ReadEngine::ReadEngine(std::string graphPath)
{
    graphPath = graphPath;
    srcPath = graphPath + "srcList.bin";
    dstPath = graphPath + "dstList.bin";
    trainMaskPath = graphPath + "trainIDs.bin";

    srcFd = open(srcPath.c_str(), O_RDONLY);
    dstFd = open(dstPath.c_str(), O_RDONLY);
    tidsFd = open(trainMaskPath.c_str(), O_RDONLY);
    if ((srcFd == -1) || (dstFd == -1) || (tidsFd == -1) ) {
        perror("open");
    }

    struct stat sb;
    if (fstat(srcFd, &sb) == -1) {
        perror("fstat");close(srcFd);
    }
    srcLength = sb.st_size;
    if (fstat(dstFd, &sb) == -1) {
        perror("fstat");close(dstFd);
    }
    dstLength = sb.st_size;
    if (fstat(tidsFd, &sb) == -1) {
        perror("fstat");close(tidsFd);
    }
    tidsLength = sb.st_size;
    edgeNUM = srcLength / sizeof(int64_t);
}

int ReadEngine::readline(std::pair<int64_t, int64_t> &edge) {
    if (readPtr == edgeNUM){
        unmapBlock(srcAddr,chunkSize);
        unmapBlock(dstAddr,chunkSize);
        return -1;
    }
        
    if (readPtr % batch == 0){
        if (chunkSize != 0) {
            unmapBlock(srcAddr,chunkSize);
            unmapBlock(dstAddr,chunkSize);
        }
        loadingMmapBlock();
    }
    edge.first = srcAddr[readPtr%batch];
    edge.second = dstAddr[readPtr%batch];
    readPtr++;
    return 0;
}

int ReadEngine::readlines(std::vector<std::pair<int64_t, int64_t>> &edges,std::vector<int64_t>& eids,int& edgesNUM) {
    if (readPtr == edgeNUM)
        return -1;
    std::pair<int64_t, int64_t> edge;
    int i = 0;
    for ( ; i < edgesNUM ; i++) {
        if(-1 != this->readline(edge)) {
            eids[i] = this->readPtr - 1;
            edges[i] = edge;
        }
    }
    return i;
    
}

void ReadEngine::loadingMmapBlock() {
    chunkSize = std::min((long)readSize, srcLength - offset);
    srcAddr = static_cast<int64_t*>(mmap(nullptr, chunkSize, PROT_READ, MAP_SHARED, srcFd, offset));
    dstAddr = static_cast<int64_t*>(mmap(nullptr, chunkSize, PROT_READ, MAP_SHARED, dstFd, offset));
    if((srcAddr == MAP_FAILED) || (dstAddr == MAP_FAILED))
    {
        perror("mmap");
        close(srcFd);close(dstFd);close(tidsFd);
    }
    offset += chunkSize;
}

void ReadEngine::unmapBlock(int64_t* addr, off_t size) {
    munmap(addr, size);
}

void ReadEngine::readTrainIdx(std::vector<int64_t>& ids) {
    int tidsNUM = tidsLength / sizeof(int64_t);
    tidsAddr = static_cast<int64_t*>(mmap(nullptr, tidsLength, PROT_READ, MAP_SHARED, tidsFd, 0));
    ids.clear();
    ids.resize(tidsNUM);
    for (int i = 0; i < tidsNUM; ++i) {
        ids[i] = tidsAddr[i];
    }
}

TGEngine::TGEngine() {}

TGEngine::TGEngine(std::string graphPath,int nodeNUM,int edgeNUM) {
    this->graphPath = graphPath;

    Fd = open(this->graphPath.c_str(), O_RDONLY);
    if (Fd == -1) {
        perror("open");
    }

    struct stat sb;
    if (fstat(Fd, &sb) == -1) {
        perror("fstat");close(Fd);
    }
    edgeLength = sb.st_size;
    this->edgeNUM = edgeNUM;
    this->nodeNUM = nodeNUM;
}

void TGEngine::loadingMmapBlock() {
    chunkSize = std::min((long)readSize, edgeLength - offset);
    edgeAddr = static_cast<int*>(mmap(nullptr, chunkSize, PROT_READ, MAP_SHARED, Fd, offset));
    if(edgeAddr == MAP_FAILED)
    {
        perror("mmap");
        close(Fd);
    }
    offset += chunkSize;
}

void TGEngine::unmapBlock(int* addr, off_t size) {
    munmap(addr, size);
}

int TGEngine::readline(std::pair<int, int> &edge) {
    if (readPtr == edgeNUM){
        unmapBlock(edgeAddr,chunkSize);
        return -1;
    }
        
    if (readPtr % batch == 0){
        if (chunkSize != 0) {
            unmapBlock(edgeAddr,chunkSize);
        }
        loadingMmapBlock();
    }
    edge.first = edgeAddr[readPtr%batch];
    edge.second = edgeAddr[readPtr%batch+1];
    readPtr += 2;
    return 0;
}

void TGEngine::convert2bin(std::string raw_graphPath,std::string new_graphPath,char delimiter,bool saveDegree) {
    std::ifstream inputFile(raw_graphPath);
    if (!inputFile.is_open()) {
        std::cerr << "Failed to open input file: " << raw_graphPath << std::endl;
        return;
    }

    std::ofstream outputFile(new_graphPath, std::ios::binary);
    if (!outputFile.is_open()) {
        std::cerr << "Failed to open output file: " << new_graphPath << std::endl;
        return;
    }

    std::string line;
    while (std::getline(inputFile, line)) {
        std::istringstream iss(line);
        int src, dst;
        char block;
        if (delimiter == ' ')
            iss >> src >> dst;
        else
            iss >> src >> block >> dst;
        std::cout << src << " --> " << dst << std::endl;
        // 将源和目标值写入二进制文件
        // outputFile.write((char *)&src, sizeof(int));
        // outputFile.write((char *)&dst, sizeof(int));
    }

    inputFile.close();
    outputFile.close();
}












