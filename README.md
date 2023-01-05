# RunAI

Run AI allows you to run a threaded Stable Diffusion socket server.

The implementation is a low level socket server which accepts 
JSON encoded byte chunks requests (1024 byte default) and
assembles them, decodes them, processes the request and returns a
response to any connected client.

**Note:** This server is not meant to be run with Automatic1111. It is a standalone 
server which requires a socket client that can communicate with the server.

You can use the **Automatic1111 webui** as a client, but it would require
middleware to be written to handle the socket connection and communication.
This could likely be done with a simple python script.

See the [Stable Diffusion directory structure section](#stable-diffusion-directory-structure) for more information.

---

The server users diffusers. On first run it will connect to huggingface.co and
download all the models required to run a given diffuser and store them in the
huggingface cache directory. This can take a while depending on your internet
connection.

---

## Client

For an example client, take a look at [krita_stable_diffusion connect.py file](https://github.com/w4ffl35/krita_stable_diffusion/blob/master/krita_stable_diffusion/connect.py) which uses this server.

---

## Features

- Offline friendly - works completely locally with no internet connection
- **Sockets**: handles byte chunks of an arbitrary size
- **Threaded**: asynchronously handle requests and responses
- **Queue**: requests and responses are handed off to a queue
- **Auto-shutdown**: server automatically shuts down after client disconnects
- Does not save images or logs to disc

## Limitations

- Only handles a single client
- Data between server and client is not encrypted
- Only uses float16 (half floats)

## Planned changes

- Encrypted sqlite database to store generated images and request parameters (optional)
- Handle multiple client connections
- Add support for upscaling and other missing diffusers functions

---

## Design choices

The choice was made to create a socket server without the use of an existing
framework because I wanted to keep the

---

## Request structure

Client makes request to a RunAI server by

1. Establishing a socket connection to the server at URL:PORT
2. Formats JSON request as a byte string
3. Sends byte-sized chunks to server
4. Signals end of message (EOM) using a specific encoded message

The server will handle this message by first enqueuing the request, then handling requests in queue by calling stable 
diffusion on each request, enqueuing the request and passing those back to the client in the same format
received.

It is up to the client to reassemble the chunks, decode the byte strine to JSON 
and handle the message.

## Stable Diffusion directory structure


This is the recommended and default setup for runai

### Linux

Default directory structure for runai Stable Diffusion

#### Base models

These models are required to run Stable Diffusion

- **CLIP** files for CLIP
- **CompVis** safety checker model (used for NSWF filtering)
- **openai** clip-vit-large-patch14 model

```
 ├── ~/stablediffusuion
    ├── CLIP
    ├── CompVis
    │   ├── stable-diffusion-safety-checker
    ├── openai
        ├── clip-vit-large-patch14
```

#### Diffusers models

These are the base models to run a particular version of Stable Diffusion.

- **runwayml**: Base models for Stable Diffusion v1
- **stabilityai**: Base models for Stable Diffusion v2

```
├── ~/stablediffusuion
   ├── runwayml
      ├── stable-diffusion-inpainting
      ├── stable-diffusion-v1-5
   ├── stabilityai
      ├── stable-diffusion-2-1-base
      ├── stable-diffusion-2-inpainting
```

#### Custom models

- **v1** should be a directory containing models using stable diffusion v1
- **v2** should be a directory containing models using stable diffusion v2

You may place diffusers folders, ckpt and safetensor files in these directories.

```
├── ~/stablediffusuion
   ├── v1
   │   ├── <folder> (diffusers directory)
   │   ├── <file>.ckpt
   │   ├── <file>.safetensor
   ├── v2
       ├── <folder> (diffusers directory)
       ├── <file>.ckpt
       ├── <file>.safetensor
```

### Automatic1111 existing files

If you are using **Automatic1111** you can place your checkpoints in the
webui models folder as you typically would, however the directory structure
which includes v1 models separated from v2 models is required for now.

This allows you to use the same checkpoints for both **Automatic1111 webui**
and this server.

For example, if your `webui` directory looks like this

```
├── /home/USER/stable-diffusion-webui/models/Stable-diffusion
    ├── <some_checkpoint_file>.ckpt
    ├── <some_other_checkpoint_file>.ckpt
    ├── <some_other_checkpoint_file_v2>.ckpt
```

You would reorganize it like this:

```
├── /home/USER/stable-diffusion-webui/models/Stable-diffusion
    ├── v1
       ├── <some_checkpoint_file>.ckpt
       ├── <some_other_checkpoint_file>.ckpt
    ├── v2
       ├── <some_other_checkpoint_file_v2>.ckpt
```

You would then set BASE_DIR to `/home/USER/stable-diffusion-webui/models/Stable-diffusion`

---

## Development Installation

### Requirements

- git
- conda
- a cuda capable GPU

1. [Install CUDA Toolkit 11.7](https://developer.nvidia.com/cuda-11-7-0-download-archive?target_os=Linux&target_arch=x86_64)
2. [Install miniconda](https://docs.conda.io/en/latest/miniconda.html)
3. Activate environment ``
4. conda activate runai
5. Install requirements `pip install -r requirements.txt`

### Installation notes

xformers increases installation time considerably; it may feel like your system 
is hanging when you see this message `Building wheel for xformers (setup.py)

### Build

First install `pyinstaller`

`pip install pyinstaller`

Then build the executable

```
./bin/buildlinux.sh
```

Test

```
cd ./dist/runai
./runai
```

This should start a server. 

[Connect a client to see if it is working properly](https://github.com/w4ffl35/krita_stable_diffusion)

---

## Building a standalone server

A standalone server can be built using the following instructions.

1. Follow [Development Installation](#development-installation) instructions
2. Install pyinstaller `pip install pyinstaller`
3. Run `build\<OS>\run`
4. The standalone server will be in the `dist` directory

---

## Running the server

`python server.py`

The following flags and options are available

- `--port` (int) - port to run server on
- `--host` (str) - host to run server on
- `--timeout` - whether to timeout after failing to receive a client connection, pass this flag for true, otherwise the server will not timeout.
- `--chunk-size` (int) - size of byte chunks to transmit to and from the client
- `--model-base-path` (str) - base directory for checkpoints
- `--max-client-connections` (int) - maximum number of client connections to accept

Example

```
python server.py --port 8080 --host https://0.0.0.0 --timeout
```

This will start a server listening on https://0.0.0.0:8080 and will timeout 
after a set number of attempts when no failing to receive a client connection.

### Request structure

Requets are sent to the server as a JSON encoded byte string. The JSON object
should look as follows

```
{
    TODO
}
```

### Model loading

The server does not automatically load a model. It waits for the client to send 
a request which contains a model path and name. The server will determine which
version of stable diffusion is in use and  which model has been selected 
to generate images. It will also determine the best model to load based on
the list of available types in the directory provided.
