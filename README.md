# RunAI

Run AI allows you to run a threaded Stable Diffusion socket server.

The implementation is a low level socket server which accepts 
JSON encoded byte chunks (by default 1024 bytes) requests and
assembles them, decodes them, processes the request and returns a
response to any connected client.

**Note:** This server is not meant to be run with Automatic1111. It is a standalone 
server which requires a socket client that can communicate with the server.

You can use the **Automatic1111 webui** as a client, but it would require
middleware to be written to handle the socket connection and communication.
This could likely be done with a simple python script.

See the [Stable Diffusion directory structure section](#stable-diffusion-directory-structure) for more information.

## Features

- Offline friendly - works completely locally with no internet connection
- **Sockets**: handles byte chunks of an arbitrary size
- **Threaded**: asynchronously handle requests and responses
- **Queue**: requests and responses are handed off to a queue
- **Auto-shutdown**: server automatically shuts down after client disconnects
- Does not save images or logs to disc

## Limitations

- Only handles a single client
- Byte chunk size is hard coded
- No-client timeout is hardcoded
- Hardcoded port
- Hardcoded URL

## Planned changes

- Allow server to start with command line options
  - host (string)
  - port (int)
  - auto_shutdown (flag)
  - message_size (int)
  - client_timeout_time (int)
- Encrypted sqlite database to store generated images and request parameters (optional)

---

## Design choices

The choice was made to create a socket server without the use of an existing
framework because I wanted to keep the

---

## Request structure

Client makes request to a RunAI server by

1. Establishing a socket connection to the server at URL:PORT
2. Formats JSON request as a byte string
3. Sends 1024 byte-sized chunk to server
4. Signals end of message (EOM) using a specific encoded message (1024 empty bits x00)

The server will handle this message by first enqueuing the request, then handling requests in queue by calling stable 
diffusion on each request, enqueuing the request and passing those back to the client in the same format
received (1024 byte sized chunks with an EOM to the client)

It is up to the client to reassemble the chunks, decode the byte strine to JSON 
and handle the message.

## Stable Diffusion directory structure


This is the recommended and default setup for runai

### Linux

Default directory structure for runai Stable Diffusion

```
├── /home/<USER>/
│   ├── stablediffusuion
│      ├── checkpoints
│         ├── v1
│         │   ├── <file(diffusers direcotry)>
│         │   ├── <file>.ckpt
│         │   ├── <file>.safetensor
│         ├── v2
│         │   ├── <file(diffusers direcotry)>
│         │   ├── <file>.ckpt
│         │   ├── <file>.safetensor
│         ├── CLIP
│         ├── CompVis
│         ├── openai
```

If you are using **Automatic1111** you can place your checkpoints in the
webui models folder as you typically would, however the directory structure
which includes v1 models separated from v2 models is required for now.

This allows you to use the same checkpoints for both **Automatic1111 webui**
and this server.

For example, if your `webui` directory looks like this

```
├── /home/USER/stable-diffusion-webui/models/Stable-diffusion
│   ├── <some_checkpoint_file>.ckpt
│   ├── <some_other_checkpoint_file>.ckpt
│   ├── <some_other_checkpoint_file_v2>.ckpt
```

You would reorganize it like this:

```
├── /home/USER/stable-diffusion-webui/models/Stable-diffusion
│   ├── v1
│      ├── <some_checkpoint_file>.ckpt
│      ├── <some_other_checkpoint_file>.ckpt
│   ├── v2
│      ├── <some_other_checkpoint_file_v2>.ckpt
```

You would then set BASE_DIR to `/home/USER/stable-diffusion-webui/models/Stable-diffusion`

---

## Development Installation

1. [Install CUDA Toolkit 11.7](https://developer.nvidia.com/cuda-11-7-0-download-archive?target_os=Linux&target_arch=x86_64)
2. [Install miniconda](https://docs.conda.io/en/latest/miniconda.html)
3. Activate environment ``
4. conda activate runai
5. Install requirements `pip install -r requirements.txt`

### Installation information

xformers increases installation time considerably; it may feel like your system 
is hanging when you see this message `Building wheel for xformers (setup.py)

--

## Running the server

`python server.py`

The server will shutdown automatically if no client connects.

The server does not automatically load a model. It waits for the client to send 
a request which contains a model path and name. The server will determine which
version of stable diffusion is in use and  which model has been selected 
to generate images. It will also determine the best model to load based on
the list of available types in the directory provided.