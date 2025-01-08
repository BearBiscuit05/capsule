#!/bin/bash
container_prefix="sgnn_node_" 
for container_id in $(docker ps -q); do
    container_name=$(docker inspect -f '{{.Name}}' "$container_id")
    if [[ "$container_name" == *"/$container_prefix"* ]]; then
        echo "Stopping container with ID: $container_id"
        docker stop "$container_id"
    fi
done

for container_id in $(docker ps -aq); do
    container_name=$(docker inspect -f '{{.Name}}' "$container_id")
    if [[ "$container_name" == *"/$container_prefix"* ]]; then
        echo "Restarting container with ID: $container_id"
        docker start "$container_id"
    fi
done