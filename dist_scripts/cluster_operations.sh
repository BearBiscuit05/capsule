#!/bin/bash
# set -e
container_ids=($(docker ps | grep "Capsule" | awk '{print $1}'))

command_to_run="conda uninstall -y dgl"

for container_id in "${container_ids[@]}"; do
    docker exec -it "$container_id" $command_to_run
done
