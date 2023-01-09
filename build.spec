# -*- mode: python ; coding: utf-8 -*-
import os
import shutil
from PyInstaller.utils.hooks import copy_metadata

block_cipher = None

datas = []
datas += copy_metadata('tqdm')
datas += copy_metadata('regex')
datas += copy_metadata('requests')
datas += copy_metadata('packaging')
datas += copy_metadata('filelock')
datas += copy_metadata('numpy')
datas += copy_metadata('tokenizers')
datas += copy_metadata('ftfy')
datas += copy_metadata('rich')
datas += copy_metadata('transformers')

# copy files
datas += [(os.path.join('VERSION'), '.')]

package_path = os.path.join("/usr/local/lib/python3.10/dist-packages/")

a = Analysis(
    [
        './server.py',
    ],
    pathex=[
        os.path.join(package_path , "torch/lib"),
        os.path.join(package_path , "torch/jit"),
    ],
    binaries=[
        (os.path.join(package_path, "torch/lib/libtorch_global_deps.so"), "torch/lib"),
        (os.path.join(package_path, "torch/bin/torch_shm_manager"), "torch/bin"),
        (os.path.join(package_path, 'nvidia/cudnn/lib/libcudnn_ops_infer.so.8'), '.'),
        (os.path.join(package_path, 'nvidia/cudnn/lib/libcudnn_cnn_infer.so.8'), '.'),
    ],
    datas=datas,
    hiddenimports=[
        "xformers",
        "tqdm",
        "ftfy",
        "rich",
        "libpng",
        "jpeg",
        "diffusers",
        "transformers",
        "nvidia",
        "taming",
        "taming.modules",
        "taming.modules.vqvae",
        "taming.modules.vqvae.quantize",
        "clip",
        "stablediffusion",
        "torch",
        "torchvision",
        "torchvision.io",
        "torch.onnx.symbolic_opset7",
        "torch.onnx.symbolic_opset8",
        "torch.onnx.symbolic_opset9",
        "torch.onnx.symbolic_opset10",
        "torch.onnx.symbolic_opset11",
        "torch.onnx.symbolic_opset12",
        "torch.onnx.symbolic_opset14",
        "torch.onnx.symbolic_opset15",
        "torch.onnx.symbolic_opset16",
        "torch.onnx.symbolic_opset17",
        "opencv",
        "einops",
        "imwatermark",
        "omegaconf",
        "contextlib",
        "itertools",
        "pytorch_lightning",
        "huggingface_hub.hf_api",
        "huggingface_hub.repository",
        "pywt._extensions._cwt",
        "kornia",
        "inspect",
        "psutil",
        "matplotlib",
        "bitsandbytes",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher
)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='runai',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='runai',
)

# copy directories to dist
shutil.copytree(
    '/usr/local/lib/python3.10/dist-packages/transformers/',
    './dist/runai/transformers'
)