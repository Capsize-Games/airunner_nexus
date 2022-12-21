# send a post request to flask server
import requests
import random

url = 'http://localhost:5000/'
# random seed
seed = random.randint(0, 1000000)
data = {
    "prompt": "qkz cell animation of batman",
    "seed": 956369,
    "batch_size":1,
    "guidance_scale":7,
    "num_inference_steps":50,
}
response = requests.post(url, json=data)