#!/bin/bash
# set -e

base_hostname="Capsule_cluster_"
container_ids=($(docker ps | grep $base_hostname | awk '{print $1}'))
command_to_run="conda uninstall -y dgl"

for container_id in "${container_ids[@]}"; do
    docker exec -it "$container_id" $command_to_run
done
