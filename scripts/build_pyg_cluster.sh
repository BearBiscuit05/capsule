#!/bin/bash
set -e

cluster_size=4
base_hostname="Pyg_cluster_"
network_name="pyg_net"
start_ip="44.0.0."
image_name="bearbiscuit/pyg_dist:v1.0"
beg_ip=1

# create Docker network
# docker network create --driver bridge --subnet 44.0.0.0/16 --gateway 44.0.0.1 pyg_net


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

