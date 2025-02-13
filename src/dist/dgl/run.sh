#!/bin/bash

prefix_path="/Capsule"

python3 ./launch.py \
--workspace ${prefix_path}/src/dist/dgl \
--num_trainers 1 \
--num_samplers 0 \
--num_servers 1 \
--part_config ${prefix_path}/data/random_pd/ogb-product.json \
--ip_config ${prefix_path}/ip_config.txt \
"/miniconda3/envs/capsule/bin/python3 graphsage.py --graph_name ogb-product --ip_config /Capsule/ip_config.txt --num_epochs 20 --batch_size 1000 --num_gpus 1"