from setuptools import setup, find_packages

setup(
    name="airunner_nexus",
    version="v1.4.4",
    author="Capsize LLC",
    description="Run a socket server for AI models.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="ai, stable diffusion, art, ai art, stablediffusion, LLM, mistral",
    license="",
    author_email="contact@capsizegames.com",
    url="https://github.com/Capsize-Games/airunner_nexus",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.10.0",
    install_requires=[
        "numpy==1.26.4",
        # Core application dependencies
        "accelerate==0.29.2",
        "huggingface-hub==0.23.0",
        "torch==2.2.2",
        "optimum==1.19.1",

        # Stable Diffusion Dependencies
        "omegaconf==2.3.0",
        "diffusers==0.27.2",
        "controlnet_aux==0.0.8",
        "einops==0.7.0",  # Required for controlnet_aux
        "Pillow==10.3.0",
        "pyre-extensions==0.0.30",
        "safetensors==0.4.3",
        "compel==2.0.2",
        "tomesd==0.1.3",

        # LLM Dependencies
        "transformers==4.40.1",
        "auto-gptq==0.7.1",
        "bitsandbytes==0.43.1",
        "datasets==2.18.0",
        "sentence_transformers==2.6.1",
        "pycountry==23.12.11",
        "sounddevice==0.4.6",  # Required for tts and stt
        "pyttsx3==2.90",  # Required for tts
        "peft==0.12.0",

        # Pyinstaller Dependencies
        "ninja==1.11.1.1",
        "JIT==0.2.7",
        # "opencv-python-headless==4.9.0.80",
        "setuptools==69.5.1",

        # Llama index
        "llama-index==0.10.32",
        "llama-index-readers-file==0.1.19",
        "llama-index-readers-web==0.1.13",
        "llama-index-embeddings-huggingface==0.2.3",
        "llama-index-llms-huggingface==0.2.6"
    ],
    dependency_links=[],
)
