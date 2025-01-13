#!/bin/bash

# container_prefix="Glt_cluster_"  
# container_prefix="Capsule_cluster_"
# container_prefix="Pyg_cluster_"  

# Check if the script received the required arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <container_prefix> <action>"
    echo "Actions: Restart, Delete"
    exit 1
fi

# Assign arguments to variables
container_prefix=$1
action=$2

# Validate action
if [[ "$action" != "Restart" && "$action" != "Delete" ]]; then
    echo "Invalid action. Please use 'Restart' or 'Delete'."
    exit 1
fi

# Perform the selected action
case $action in
    Restart)
        echo "Restarting all containers with prefix: $container_prefix"
        for container_id in $(docker ps -aq --filter "name=$container_prefix"); do
            docker restart "$container_id"
        done
        ;;
    Delete)
        echo "Deleting all containers with prefix: $container_prefix"
        for container_id in $(docker ps -aq --filter "name=$container_prefix"); do
            docker rm -f "$container_id"
        done
        ;;
    *)
        echo "Invalid action. Please use 'Restart' or 'Delete'."
        exit 1
        ;;
esac

echo "Operation completed."
