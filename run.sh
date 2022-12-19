export DISABLE_TELEMETRY=true
export HF_ENDPOINT=""
export HF_HUB_OFFLINE=true


# add /home/joe/Projects/ai/runai2/diffusers to python include path
export PYTHONPATH=$PYTHONPATH:/home/joe/Projects/ai/runai2/diffusers

/home/joe/miniconda3/envs/runai2/bin/python server.py