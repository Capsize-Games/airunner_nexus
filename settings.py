"""
Default settings for the project.
"""
MAX_CLIENTS = 1
ACTIONS = {
    "txt2img": 0,
    "img2img": 1,
    "GFPGAN": 2,
    "RemoveBackground": 3,
    "CancelImage": 4,
    "Inpainting": 5,
    "Refine": 6,
    "Colorize": 7,
    "Upscale": 8,
}
STABLE_DIFFUSION_DIRECTORY = ""
DEBUG = True
PACKET_SIZE = 1024
DEFAULT_PORT = 50006
DEFAULT_HOST = "0.0.0.0"