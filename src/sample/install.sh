#！/bin/bash
rm -rf build
rm -rf dist
rm -rf signn.egg-info
python3 test_torch/setup.py install