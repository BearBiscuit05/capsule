#!/bin/bash
set -e

# 参数设置
cluster_size=4
base_hostname="sgnn_node_"
network_name="sgnn_net"
start_ip="43.0.0."  # 起始IP地址
image_name="sgnn:v2.0"
deviceId=0
beg_ip=8

raw_path=/home/bear/workspace/single-gnn
# data_path=/home/bear/workspace/data
dgl_path=/home/bear/workspace/dgl_0.9


if ! docker network inspect $network_name &> /dev/null; then
    docker network create --driver bridge --subnet 43.0.0.0/16 --gateway 43.0.0.1 $network_name
fi


for i in $(seq 0 "$((cluster_size - 1))"); do
    hostname="$base_hostname$i"
    ip_octet=$(($beg_ip+i))
    ip="$start_ip$ip_octet"

    if docker ps -a | grep -q "$hostname"; then
        echo "Container '$hostname' exists."
        docker stop $hostname
        docker rm $hostname
    fi

    docker run -d -it \
        --ulimit memlock=-1 \
        --name "$hostname" \
        --ulimit nofile=65535 \
        --gpus="device=$i" \
        --ipc=host \
        --hostname "$hostname" \
        -v $raw_path:/sgnn \
        -v $dgl_path:/dgl \
        --net $network_name \
        --ip $ip \
        $image_name /bin/bash -c "service ssh restart && /bin/bash"

    echo "Container $hostname created with IP: $ip"
done
