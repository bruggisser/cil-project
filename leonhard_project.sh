module load gcc/6.3.0 python_gpu/3.8.5
module load eth_proxy
python -m venv ./venv
source venv/bin/activate
pip3 install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:~/project
