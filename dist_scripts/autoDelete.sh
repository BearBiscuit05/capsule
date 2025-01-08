#!/bin/bash

container_prefix="Capsule_cluster_"  # 替换为你希望的前缀

for container_id in $(docker ps -q); do
    container_name=$(docker inspect -f '{{.Name}}' "$container_id")
    
    if [[ "$container_name" == *"/$container_prefix"* ]]; then
        echo "Stopping and removing container with ID: $container_id"
        docker stop "$container_id"
        docker rm "$container_id"
    fi
done

for container_id in $(docker ps -aq); do
    container_name=$(docker inspect -f '{{.Name}}' "$container_id")

    # 检查容器名称是否包含指定前缀
    if [[ "$container_name" == *"/$container_prefix"* ]]; then
        echo "Removing stopped container with ID: $container_id"
        docker rm "$container_id"
    fi
done
