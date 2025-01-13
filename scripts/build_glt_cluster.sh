#!/bin/bash
set -e

cluster_size=2
base_hostname="Glt_cluster_"
network_name="glt_net"
start_ip="45.0.0."  # 起始IP地址i
image_name="bearbiscuit/glt:v2.0"
beg_ip=1

# create Docker network
# docker network create --driver bridge --subnet 45.0.0.0/16 --gateway 45.0.0.1 glt_net


for i in $(seq 1 "$cluster_size"); do
    hostname="$base_hostname$i"
    ip_octet=$(($beg_ip+i))
    ip="$start_ip$ip_octet"
    custom_hosts_params+=" --add-host $hostname:$ip"
done


for i in $(seq 1 "$cluster_size"); do
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
        --gpus="device=$((i-1))" \
        --hostname "$hostname" \
        --shm-size 32G \
        -v /home/bear/workspace/capsule:/Capsule \
        -v /home/bear/workspace/baseline_dist:/baseline_dist \
        --net $network_name \
        --ip $ip \
        $image_name \
        /bin/bash -c "service ssh restart && /bin/bash"
    
    echo "Container $base_hostname$i created with hostname $hostname with IP:$ip"
done

# python -m pip install --upgrade pip -i https://mirrors.ustc.edu.cn/pypi/simple
# pip install -i https://mirrors.ustc.edu.cn/pypi/simple graphlearn-torch ogb pyyaml pycrypto paramiko click
# python partition_ogbn_dataset.py --dataset=ogbn-products --num_partitions=2
