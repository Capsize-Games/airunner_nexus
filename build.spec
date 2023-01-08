# -*- mode: python ; coding: utf-8 -*-
import os
import sys ; sys.setrecursionlimit(sys.getrecursionlimit() * 5)
from PyInstaller.utils.hooks import copy_metadata

block_cipher = None

datas = [
]
datas += copy_metadata('tqdm')
datas += copy_metadata('regex')
datas += copy_metadata('requests')
datas += copy_metadata('packaging')
datas += copy_metadata('filelock')
datas += copy_metadata('numpy')
datas += copy_metadata('tokenizers')
datas += copy_metadata('ftfy')
datas += copy_metadata('rich')


a = Analysis(
    [
        './server.py',
    ],
    pathex=[
    ],
    binaries=[
    ],
    datas=datas,
    hiddenimports=[
        "xformers",
        "krita",
        "tqdm",
        "ftfy",
        "rich",
        "libpng",
        "jpeg",
        "diffusers",
        "transformers",
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
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

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