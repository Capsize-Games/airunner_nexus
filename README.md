# RunAI

Run AI allows you to run a LLMs using a socket server.

---

## Features

- Offline friendly - works completely locally with no internet connection (must first download models)
- **Sockets**: handles byte packets of an arbitrary size
- **Threaded**: asynchronously handle requests and responses
- **Queue**: requests and responses are handed off to a queue

---

## Limitations

### Data between server and client is not encrypted

This only matters if someone wants to create a production ready version of this
server which would be hosted on the internet. This server is not designed for
that purpose. It was designed with a single use-case in mind: the ability to run
Stable Diffusion (and other AI models) locally. It was designed for use with the
Krita Stable Diffusion plugin, but can work with any interface provided someone 
writes a client for it.

### Only works with Mistral

This library was designed to work with the Mistral model, but it can be expanded
to work with any LLM.

---

## Installation

```bash
pip install runai
cp src/runai/default.settings.py src/runai/settings.py
```

Modify `settings.py` as you see fit.

---

## Run server and client

See `src/runai/server.py` for an example of how to run the server and `src/runai/client.py` for an example of how to run 
the client. Both of these files can be run directly from the command line.

The socket client will continuously attempt to connect to the server until it is successful. The server will accept
connections from any client on the given port.
