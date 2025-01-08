#!/bin/bash
hostname=sgnn_nodebear
deviceId=0
image_name=dist_sgnn:v2.0

# 定义工作目录
raw_path=/home/bear/workspace/single-gnn
data_path=/home/bear/workspace/data
dgl_path=/home/bear/workspace/signn_dgl_0.9

# 运行 Docker 容器
docker run -d -it \
    --ulimit memlock=-1 \
    --name "$hostname" \
    --ulimit nofile=65535 \
    --gpus="device=$deviceId" \
    -v $raw_path:/sgnn \
    -v $data_path:/data \
    -v $dgl_path:/dgl \
    $image_name /bin/bash -c "service ssh restart && /bin/bash"
