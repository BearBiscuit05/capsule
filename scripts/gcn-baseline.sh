#!/bin/bash
cd ../src/train/dgl
python gcn.py --mode mixed --fanout [10,25] --layers 2 --dataset ogb-products
python gcn.py --mode mixed --fanout [5,10,15] --layers 3 --dataset ogb-products
python gcn.py --mode mixed --fanout [10,15] --layers 2 --dataset ogb-products
python gcn.py --mode mixed --fanout [10,10,10] --layers 3 --dataset ogb-products
python gcn.py --mode mixed --fanout [10,25] --layers 2 --dataset Reddit
python gcn.py --mode mixed --fanout [5,10,15] --layers 3 --dataset Reddit
python gcn.py --mode mixed --fanout [10,15] --layers 2 --dataset Reddit
python gcn.py --mode mixed --fanout [10,10,10] --layers 3 --dataset Reddit

cd ../pyg
python gcn.py --fanout [25,10] --layers 2 --dataset ogb-products
python gcn.py --fanout [15,10,5] --layers 3 --dataset ogb-products
python gcn.py --fanout [15,10] --layers 2 --dataset ogb-products
python gcn.py --fanout [10,10,10] --layers 3 --dataset ogb-products
python gcn.py --fanout [25,10] --layers 2 --dataset Reddit
python gcn.py --fanout [15,10,5] --layers 3 --dataset Reddit
python gcn.py --fanout [15,10] --layers 2 --dataset Reddit
python gcn.py --fanout [10,10,10] --layers 3 --dataset Reddit

cd ../sgnn
python ../../../config/configChange.py --pattern "../../../config/dgl_products_gcn.json" --key fanout --value [25,10]
python dgl_gcn.py --mode mixed --json_path ../../../config/dgl_products_gcn.json
python ../../../config/configChange.py --pattern "../../../config/dgl_products_gcn.json" --key fanout --value [15,10,5]
python dgl_gcn.py --mode mixed --json_path ../../../config/dgl_products_gcn.json
python ../../../config/configChange.py --pattern "../../../config/dgl_products_gcn.json" --key fanout --value [15,10]
python dgl_gcn.py --mode mixed --json_path ../../../config/dgl_products_gcn.json
python ../../../config/configChange.py --pattern "../../../config/dgl_products_gcn.json" --key fanout --value [10,10,10]
python dgl_gcn.py --mode mixed --json_path ../../../config/dgl_products_gcn.json

python ../../../config/configChange.py --pattern "../../../config/dgl_reddit_8.json" --key fanout --value [25,10]
python dgl_gcn.py --mode mixed --json_path ../../../config/dgl_reddit_8.json
python ../../../config/configChange.py --pattern "../../../config/dgl_reddit_8.json" --key fanout --value [15,10,5]
python dgl_gcn.py --mode mixed --json_path ../../../config/dgl_reddit_8.json
python ../../../config/configChange.py --pattern "../../../config/dgl_reddit_8.json" --key fanout --value [15,10]
python dgl_gcn.py --mode mixed --json_path ../../../config/dgl_reddit_8.json
python ../../../config/configChange.py --pattern "../../../config/dgl_reddit_8.json" --key fanout --value [10,10,10]
python dgl_gcn.py --mode mixed --json_path ../../../config/dgl_reddit_8.json

python ../../../config/configChange.py --pattern "../../../config/pyg_products_gcn.json" --key fanout --value [25,10]
python pyg_gcn.py --mode mixed --json_path ../../../config/pyg_products_gcn.json
python ../../../config/configChange.py --pattern "../../../config/pyg_products_gcn.json" --key fanout --value [15,10,5]
python pyg_gcn.py --mode mixed --json_path ../../../config/pyg_products_gcn.json
python ../../../config/configChange.py --pattern "../../../config/pyg_products_gcn.json" --key fanout --value [15,10]
python pyg_gcn.py --mode mixed --json_path ../../../config/pyg_products_gcn.json
python ../../../config/configChange.py --pattern "../../../config/pyg_products_gcn.json" --key fanout --value [10,10,10]
python pyg_gcn.py --mode mixed --json_path ../../../config/pyg_products_gcn.json

python ../../../config/configChange.py --pattern "../../../config/pyg_reddit_8.json" --key fanout --value [25,10]
python pyg_gcn.py --mode mixed --json_path ../../../config/pyg_reddit_8.json
python ../../../config/configChange.py --pattern "../../../config/pyg_reddit_8.json" --key fanout --value [15,10,5]
python pyg_gcn.py --mode mixed --json_path ../../../config/pyg_reddit_8.json
python ../../../config/configChange.py --pattern "../../../config/pyg_reddit_8.json" --key fanout --value [15,10]
python pyg_gcn.py --mode mixed --json_path ../../../config/pyg_reddit_8.json
python ../../../config/configChange.py --pattern "../../../config/pyg_reddit_8.json" --key fanout --value [10,10,10]
python pyg_gcn.py --mode mixed --json_path ../../../config/pyg_reddit_8.json