import json
import signal
import socket
import threading
import time
from typing import Optional
import flask
import io

import numpy as np
import torch
import queue
from PIL import Image
import settings
from diffusers import StableDiffusionPipeline
from pytorch_lightning import seed_everything
from logger import logger
from exceptions import FailedToSendError, NoConnectionToClientError
import messagecodes as codes
LONG_MESSAGE_SIZE=5610434
RESPONSE_QUEUE = queue.SimpleQueue()

class SDRunner:
    _current_model = "runwayml/stable-diffusion-v1-5"

    @property
    def current_model(self):
        return self._current_model

    @current_model.setter
    def current_model(self, model):
        if self._current_model != model:
            self._current_model = model
            self.load_model()

    def load_model(self):
        torch.cuda.empty_cache()
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.current_model,
            torch_dtype=torch.half,
            # revision="fp16"
        )
        self.pipe.skip_nsfw = True
        self.pipe.enable_xformers_memory_efficient_attention()
        self.pipe.to("cuda")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_model()

    def generator_sample(self, data, image_handler):
        self.image_handler = image_handler
        return self.generate(data)

    def generate(self, data):
        options = data["options"]

        # get model and switch to it
        self.current_model = options.get("model", self.current_model)

        # Get the other values from options
        seed = int(options.get("seed", 42))
        guidance_scale = float(options.get("scale", 7.5))
        num_inference_steps = int(options.get("ddim_steps", 50))
        negative_prompt = options.get("negative_prompt", "")
        do_nsfw_filter = bool(options.get("do_nsfw_filter", False))
        do_watermark = bool(options.get("do_watermark", False))
        prompt = options.get("prompt", "")
        C = int(options.get("C", 4))
        f = int(options.get("f", 8))
        batch_size = int(data.get("n_samples", 1))

        # sample the model
        with torch.no_grad() as _torch_nograd, \
            torch.cuda.amp.autocast() as _torch_autocast:
            # try:
                # clear cuda cache
                for n in range(0, batch_size):
                    seed = seed + n
                    filename = seed
                    seed_everything(seed)
                    image = self.pipe(
                        prompt,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps
                    ).images[0]

                    # use pillow to convert the image to a byte array
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format='PNG')
                    img_byte_arr = img_byte_arr.getvalue()
                    #return flask.Response(img_byte_arr, mimetype='image/png')
                    self.image_handler(img_byte_arr, data)
            # except Exception as e:
            #     print("Error during generation 1")
            #     print(e)
            #     #return flask.jsonify({"error": str(e)})

class Server(flask.Flask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        torch.cuda.empty_cache()
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.half,
            #revision="fp16"
        )
        self.pipe.skip_nsfw = True
        # enable_xformers_memory_efficient_attention
        self.pipe.enable_xformers_memory_efficient_attention()
        self.pipe.to("cuda")

        # add endpoints
        self.add_url_rule('/', 'generate', self.generate, methods=['POST'])
        self.add_url_rule('/switch-model', 'switch_model', self.switch_model, methods=['POST'])

    def run(self, *args, **kwargs):
        super().run(*args, **kwargs)

    # create endpoint
    def generate(self):
        # get data from request
        data = flask.request.json

        options = data["options"]

        # set the seed
        seed = options.get("seed", 42)
        guidance_scale = options.get("scale", 7.5)
        num_inference_steps = options.get("ddim_steps", 50)
        negative_prompt = options.get("negative_prompt", "")
        do_nsfw_filter = options.get("do_nsfw_filter", False)
        do_watermark = options.get("do_watermark", False)
        prompt = options.get("prompt", "")
        C = options.get("C", 4)
        f = options.get("f", 8)
        batch_size = data.get("n_samples", 1)

        with torch.no_grad() as _torch_nograd, \
            torch.cuda.amp.autocast() as _torch_autocast:
            try:
                # clear cuda cache
                for n in range(0, batch_size):
                    seed = seed + n
                    filename = seed
                    seed_everything(seed)
                    image = self.pipe(
                        prompt,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps
                    ).images[0]

                    # use pillow to convert the image to a byte array
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format='PNG')
                    img_byte_arr = img_byte_arr.getvalue()
                    return flask.Response(img_byte_arr, mimetype='image/png')

                    #image.save(f"output/{filename}.png")
            except Exception as e:
                print("Error during generation 2")
                print(e)
                return flask.jsonify({"error": str(e)})


    def switch_model(self):
        # get data from request
        data = flask.request.json
        model_id = data["model_id"]
        torch.cuda.empty_cache()
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.half,
            revision="fp16"
        )
        self.pipe.skip_nsfw = True
        # enable_xformers_memory_efficient_attention
        self.pipe.enable_xformers_memory_efficient_attention()
        self.pipe.to("cuda")

        # return text
        return flask.jsonify({
            "output": "model switched"
        })


# create a SocketServer that does the same thing as the Flask server but uses sockets
class SocketServerOld:
    def __init__(self, host="localhost", port=5000):
        self.socket = None
        self.host = host
        self.port = port
        self.start_server()

    def start_server(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((self.host, self.port))
        print("listening on port", self.port)
        self.wait_for_connection()


    def wait_for_connection(self):
        # accept client connection
        self.conn, self.addr = self.socket.accept()
        print('Connected by', self.addr)

    def run(self):
        while True:
            data = self.conn.recv(1024)
            if not data:
                break
            print("received data:", data)
            self.conn.sendall(data)
        self.conn.close()

    def generate(self):
        # get data from request
        data = flask.request.json

        options = data["options"]

        # set the seed
        seed = options.get("seed", 42)
        guidance_scale = options.get("scale", 7.5)
        num_inference_steps = options.get("ddim_steps", 50)
        negative_prompt = options.get("negative_prompt", "")
        do_nsfw_filter = options.get("do_nsfw_filter", False)
        do_watermark = options.get("do_watermark", False)
        prompt = options.get("prompt", "")
        C = options.get("C", 4)
        f = options.get("f", 8)
        batch_size = data.get("n_samples", 1)

        with torch.no_grad() as _torch_nograd, \
                torch.cuda.amp.autocast() as _torch_autocast:
            try:
                # clear cuda cache
                for n in range(0, batch_size):
                    seed = seed + n
                    filename = seed
                    seed_everything(seed)
                    image = self.pipe(
                        prompt,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps
                    ).images[0]

                    # use pillow to convert the image to a byte array
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format='PNG')
                    img_byte_arr = img_byte_arr.getvalue()
                    self.send_message(img_byte_arr)

                    # image.save(f"output/{filename}.png")
            except Exception as e:
                print("Error during generation 3")
                print(e)
                # use send_message to send the error to the client
                self.send_message({"error": str(e)})

    def send_message(self, message):
        self.conn.sendall(message)


class Connection:
    """
    Base connection class.
    """
    threads = []

    def start_thread(
            self, target: Optional, daemon: bool = False, name: str = None
    ):
        """
        Start a thread and append it to the list of threads on this object.
        :param target: func, thread will run this function
        :param daemon: boolean, whether the thread is a daemon thread
        :param name: str, name of the thread
        return: thread
        """
        thread = threading.Thread(target=target, daemon=daemon)

        if name:
            thread.name = name
        thread.start()
        self.threads.append(thread)
        return thread

    def connect(self):
        """
        Override this method to set up a connection to something.

        Do not call connect directly, it should be used in a thread.

        Use the start() method which starts this method in a new thread.
        :return: None
        """

    def disconnect(self):
        """
        Override this method to disconnect from something.
        :return: None
        """

    def reconnect(self):
        """
        Disconnects then reconnects to service. Does not stop the thread.
        :return: None
        """
        self.disconnect()
        self.connect()

    def start(self):
        """
        Starts a new thread with a connection to service.
        :return: None
        """
        self.start_thread(
            target=self.connect,
            name="Connection thread"
        )

    def stop(self):
        """
        Disconnects from service and stops the thread
        :return: None
        """
        self.disconnect()
        logger.info("Stopping connection thread...")
        for index, thread in enumerate(self.threads):
            total = len(self.threads)
            name = thread.getName()
            logger.info(f"{index+1} of {total} Stopping thread {name}")
            try:
                thread.join()
            except RuntimeError:
                logger.info(f"Thread {thread.getName()} not running")
            logger.info(f"Stopped thread {thread.getName()}...")
        logger.info("All threads stopped")

    def __init__(self, *args, **kwargs):
        self.start()


class SocketConnection(Connection):
    """
    Opens a socket on a server and port.

    parameters:
    :host: Hostname or IP address of the service
    :port: Port of the service
    """
    port = settings.DEFAULT_PORT
    host = settings.DEFAULT_HOST
    soc = None
    soc_connection = None
    soc_addr = None

    def open_socket(self):
        """
        Open a socket connection
        :return:
        """

    def handle_open_socket(self):
        """
        Override this method to handle open socket
        :return:
        """

    def connect(self):
        """
        Open a socket and handle connection
        :return: None
        """
        self.open_socket()
        self.handle_open_socket()

    def disconnect(self):
        """
        Disconnect from socket
        :return: None
        """
        if self.soc_connection:
            self.soc_connection.close()
        self.soc.close()
        self.soc_connection = None

    def initialize_socket(self):
        """
        Initialize a socket. Use timeout to prevent constant blocking.
        :return: None
        """
        self.soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.soc.settimeout(3)

    def __init__(self, *args, **kwargs):
        """
        Initialize the socket connection, call initialize socket prior
        to calling super because super will start a thread calling connect,
        and connect opens a socket.

        Failing to call initialize socket prior to super will result in an error
        """
        self.initialize_socket()
        super().__init__(*args, **kwargs)
        self.queue = queue.SimpleQueue()


class SocketServer(SocketConnection):
    """
    Opens a socket on a server and port.
    """
    quit_event = None

    # pylint: disable=too-many-instance-attributes

    def reset_connection(self):
        """
        Reset connection to service
        :return: None
        """
        self.disconnect()
        self.initialize_socket()
        self.has_connection = False
        self.open_socket()
        self.listen_to_socket()

    def callback(self, msg):
        """
        Override this method or pass it in as a parameter to handle messages
        :param msg:
        :return:
        """

    def worker(self):
        """
        Worker is started in a thread and waits for messages that are appended
        to the queue. When a message is received, it is passed to the callback
        method. The callback method should be overridden to handle the message.
        :return:
        """

    def open_socket(self):
        """
        Open a socket connection
        :return: None
        """
        try:
            self.soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.soc.settimeout(1)
            self.soc.bind(("", 50006))
        except socket.error as err:
            logger.info(f"Failed to open a socket at {self.host}:{self.port}")
            logger.info(str(err))
        except Exception as _exc:   # pylint: disable=broad-except
            logger.error(f"Failed to open a socket at {self.host}:{self.port}")
            logger.error(_exc)
        logger.info(f"Socket opened {self.soc}")

    def listen_to_socket(self):
        """
        Listen to socket for connections
        :return: None
        """
        self.soc.listen(self.max_clients)
        logger.info(f"Listening for connections on {self.host}:{self.port}")

    def try_quit(self):
        """
        Try to quit the thread
        :return: None
        """
        if self.quit_event.is_set():
            RESPONSE_QUEUE.put(b"quit")
            if self.soc_connection:
                self.soc_connection.close()
                self.soc_connection = None
            if self.queue:
                self.queue.put("quit")

    def do_send(self, msg):
        """
        Send a message to the client
        :param msg: The message to send in bytes
        :return: None
        """
        size_in_bytes = len(msg)
        if not self.soc_connection:
            raise NoConnectionToClientError("No connection to client")
        else:
            try:
                bytes_sent = self.soc_connection.send(msg)
            except Exception as e:
                logger.error("something went wrong")
                bytes_sent = 0
                logger.error(e)
            if bytes_sent != size_in_bytes:
                raise FailedToSendError()
        return bytes_sent

    def send_msg(self, msg=None):
        """
        Send a message to the client
        :param msg: The message to send
        :return: None
        """
        success = False
        bytes_sent = 0
        try:
            bytes_sent = self.do_send(msg)
            success = True
        except FailedToSendError as e:
            # return to queue
            logger.error("failed to send connection to client")
        except NoConnectionToClientError as e:
            # return to queue
            logger.error("Lost connection to client")
        if not success:
            logger.error("failed to send message, adding back to queue")
            self.message = msg
        return bytes_sent

    @staticmethod
    def process_message(msg):
        """
        Process a message from the client
        :param msg: the binary message
        :return: the processed message
        """
        # iterate over data_struct, split msg up into bytes and build a dict
        # based on keys from DATA_STRUCT and values converted from bytes
        # to the correct type

        # processed = PROCESSED_DATA_STRUCT.copy()
        # data_struct_keys = list(DATA_STRUCT.keys())
        # for index, item_key in enumerate(data_struct_keys):
        #     val = DATA_STRUCT[item_key]
        #     size = val[0]
        #     item_type = val[1]
        #     processed[item_key] = item_type(msg[index*size:(index+1)*size])
        # return processed
        pass

    @property
    def signal_byte_size(self):
        return 1024

    def is_expected_message(self, chunk, byte):
        return chunk == byte * self.signal_byte_size

    def is_quit_message(self, chunk):
        return self.is_expected_message(chunk, b'x')

    def is_cancel_message(self, chunk):
        return self.is_expected_message(chunk, b'c')

    def is_model_switch_message(self, chunk):
        bytes = [
            {
                "byte": b'A',
                "model": "v1-4"
            },
            {
                "byte": b'B',
                "model": "v1-5"
            },
            {
                "byte": b'C',
                "model": "v1-5-inpainting"
            },
            {
                "byte": b'Z',
                "model": "custom"
            },
        ]
        selected_model = None
        for item in bytes:
            if self.is_expected_message(chunk, item["byte"]):
                selected_model = item["model"]
                break
        return selected_model


    def handle_quit_message(self):
        logger.info("Quit")
        self.quit_event.set()
        self.message = None
        RESPONSE_QUEUE.put(b"quit")

    def handle_cancel_message(self):
        logger.info("Cancel image")
        self.message = None
        self.cancel()

    def handle_model_switch_message(self, model):
        pass

    def switch_model(self, model):
        logger.info("switch_model")
        self.message = json.dumps({
            "reqtype": "switch_model",
            "model": model
        }).encode()
        # self.quit_event.set()
        # self.message = None
        #self.quit_event.set()

    def get_chunk(self):
        chunk = self.soc_connection.recv(self.signal_byte_size)
        if chunk == b'':
            raise RuntimeError("socket connection broken")
        chunk = chunk.strip(b'\x00')
        return chunk

    def handle_open_socket(self):
        # pylint: disable=too-many-statements
        # pylint: disable=too-many-branches
        """
        Listen for incoming connections.
        Returns: None
        """
        self.listen_to_socket()
        current_state = codes.AWAITING_CONNECTION
        logger.info("Waiting for connections")
        self.soc.settimeout(3)
        total_timeouts = 0
        while not self.quit_event.is_set():
            if current_state is codes.AWAITING_CONNECTION:
                try:
                    if not self.quit_event.is_set():
                        self.soc_connection, self.soc_addr = self.soc.accept()
                    if self.soc_connection:
                        total_timeouts = 0
                        self.has_connection = True
                        current_state = codes.AWAITING_MESSAGE
                        logger.info(f"connected with {self.soc_addr}")
                except socket.timeout:
                    total_timeouts += 1
                    if total_timeouts >= 3:
                        self.quit_event.set()
                        break
                except Exception as exc:  # pylint: disable=broad-except
                    logger.error(exc)

            if current_state is codes.AWAITING_MESSAGE:
                msg = None
                try:
                    chunks = []
                    bytes_recd = 0
                    while True:
                        chunk = self.get_chunk()

                        # if chunk size is 0 then it tells us the
                        # client is done sending the message
                        if len(chunk) == 0:
                            break

                        if self.is_quit_message(chunk):
                            self.handle_quit_message()
                            break

                        if self.is_cancel_message(chunk):
                            self.handle_cancel_message()
                            break

                        switch_model = self.is_model_switch_message(chunk)
                        if switch_model:
                            self.switch_model(switch_model)
                            break

                        if chunk != b'':
                            chunks.append(chunk)
                    msg = b''.join(chunks)
                except socket.timeout:
                    pass
                except AttributeError:
                    logger.error("attribute error")
                except ConnectionResetError:
                    logger.error("connection reset error")
                    current_state = codes.AWAITING_CONNECTION
                except Exception as exc:  # pylint: disable=broad-except
                    logger.error(exc)
                    current_state = codes.AWAITING_CONNECTION

                if msg is not None and msg != b'':
                    logger.info("message received")
                    self.message = msg  # push directly to queue
                    self.soc_connection.settimeout(None)
                    current_state = codes.AWAITING_MESSAGE
                else:
                    # check if connection is still valid
                    if not self.soc_connection:
                        logger.error("connection lost, invalid soc_connection")
                        current_state = codes.AWAITING_CONNECTION

                    if current_state == codes.AWAITING_CONNECTION:
                        logger.info("Connection with client lost")
                        logger.info("resetting socket")
                        self.reset_connection()

            if self.quit_event.is_set():
                break

        logger.info("server stopped")

        # time.sleep(1)

        self.stop()

    def cancel(self):
        pass

    def watch_connection(self):
        """
        Watch the connection and shutdown if the server is the connection
        is lost.
        """
        while not self.quit_event.is_set():
            # check if ctrl+c was pressed
            if self.try_quit():
                logger.info("shutting down")
                break
            time.sleep(1)

    def __init__(self, *args, **kwargs):
        self.quit_event = threading.Event()
        self.has_connection = False
        self.message = None
        RESPONSE_QUEUE = None
        if not RESPONSE_QUEUE:
            RESPONSE_QUEUE = queue.SimpleQueue()
        super().__init__(*args, **kwargs)
        self.quit_event.clear()
        self.max_clients = kwargs.get(
            "max_clients",
            settings.MAX_CLIENTS
        )
        signal.signal(signal.SIGINT, self.quit_event.set)  # handle ctrl+c
        self.start_thread(
            target=self.worker,
            name="socket server worker"
        )
        self.start_thread(
            target=self.watch_connection,
            name="watch connection"
        )


class SimpleEnqueueSocketServer(SocketServer):
    """
    Simple socket server that enqueues messages.
    """
    _failed_messages = []  # list to hold failed messages

    @property
    def message(self):
        """
        Does nothing. Only used for the setter.
        """
        return ""

    @message.setter
    def message(self, msg):
        """
        Place incoming messages onto the queue
        """
        self.queue.put(msg)

    def worker(self):
        """
        Start a worker to handle request queue
        """
        logger.info("Enqueue worker started")
        while not self.quit_event.is_set():
            if self.has_connection:
                # set timeout on queue
                try:
                    msg = self.queue.get(timeout=1)  # get a message from the queue
                    if msg is not None:
                        logger.info("Received message from queue")
                        try:  # send to callback
                            self.callback(msg)
                        except Exception as err:  # pylint: disable=broad-except
                            logger.info(f"callback error: {err}")
                            raise (err)
                except queue.Empty:
                    pass
            if self.quit_event.is_set():
                break
            time.sleep(1)
        logger.info("SERVER WORKER: worker stopped")

    def __init__(self, *args, **kwargs):
        self.do_run = True
        self.queue = queue.SimpleQueue()
        self.image_size = 512  # set this via data in request
        super().__init__(*args, **kwargs)


class StableDiffusionRequestQueueWorker(SimpleEnqueueSocketServer):
    """
    A socket server that listens for requests and enqueues them to a queue
    """
    sdrunner = None

    def callback(self, data):
        """
        Handle a stable diffusion request message
        :return: None
        """
        reqtype = None
        try:
            # convert ascii to json
            data = json.loads(data.decode("ascii"))
            # get reqtype based on action in data
            actionID = data.get("action", None)
            self.image_size = data.get("W", None)

            for k,v in settings.ACTIONS.items():
                if k == actionID:
                    reqtype = k
                    break
        except UnicodeDecodeError as err:
            logger.error(f"something went wrong with a request from the client")
            logger.error(f"UnicodeDecodeError: {err}")
            return
        except json.decoder.JSONDecodeError as err:
            logger.error(f"Improperly formatted request from client")
            return
        data["reqtype"] = reqtype
        if reqtype == "txt2img" or reqtype == "img2img":
            # iterate over data and print everything but init_img
            for k,v in data.items():
                if k != "init_img":
                    logger.info(f"{k}: {v}")
            self.sdrunner.generator_sample(data, self.handle_image)
            logger.info("Text to image sample complete")
        elif reqtype == "Inpainting":
            self.sdrunner.inpaint_sample(data, self.handle_image)
        else:
            logger.error("NO IMAGE RESPONSE")

    def handle_image(self, response, options):
        if response is not None and response != b'':
            img = response#self.prep_image(response, options)
            RESPONSE_QUEUE.put(img)

    def prep_image(self, response, options, dtype=np.uint8):
        # logger.info("sending response")
        buffered = io.BytesIO()
        Image.fromarray(response.astype(dtype), 'RGB').save(buffered)
        image = buffered.getvalue()

        if "remove_background" in options and options["remove_background"]:
            #image = rembg.remove(image)
            print("remove image background")
            pass
        #
        # # if options.gfpgan is set, run the image through gfpgan to fix faces
        # if "gfpgan" in options and options["gfpgan"]:
        #     pass

        return image

    def cancel(self):
        self.sdrunner.cancel()

    def response_queue_worker(self):
        """
        Wait for responses from the stable diffusion runner and send
        them to the client
        """
        logger.info("RESPONSE QUEUE WORKER STARTING")
        while not self.quit_event.is_set():
            res = RESPONSE_QUEUE.get()
            if res is not None and res != b'':
                if res == b"quit": break
                if res == b"cancel":
                    logger.info("response queue: cancel")
                    self.sdrunner.cancel()
                    break
                # logger.info("sending response")
                # logger.info("GOT SAMPLES FROM IMAGE GENERATOR")
                img = res
                bytes_sent = 0
                # expected_byte_size = settings.BYTE_SIZES[self.image_size]
                actual_size = len(img)
                logger.info("SENDING MESSAGE OF SIZE {}".format(actual_size))

                try:
                    # pad the image img_bytes with zeros
                    img = res  # + b'\x00' * (expected_byte_size - actual_size)
                    bytes_sent = self.send_msg(img)
                    # send a null byte
                    self.send_msg(b'\x00')

                    # logger.info(f"sent {bytes_sent} bytes")

                    # if self.soc_connection:
                    #     #self.soc_connection.settimeout(1)
                    #     pass
                    # else:
                    #     raise FailedToSendError(
                    #         # "%d > %d image too large, refusing to transmit" % (
                    #         #     actual_size, expected_byte_size
                    #         # )
                    #     )
                except FailedToSendError as ect:
                    logger.error("failed to send message")
                    logger.error(ect)
                    self.reset_connection()
            if self.quit_event.is_set(): break
            time.sleep(0.1)
        logger.info("ENDING RESPONSE WORKER")

    def stop(self):
        super().stop()
        self.quit_event.set()

    def init_sd_runner(self):
        """
        Initialize the stable diffusion runner
        return: None
        """
        logger.info("Starting Stable Diffusion Runner")

    def do_start(self):
        # create a stable diffusion runner service
        logger.info("START THE RESPONSE QUEUE WORKER")
        self.start_thread(
            target=self.response_queue_worker,
            name="response queue worker"
        )
        sd_runner_thread = self.start_thread(
            target=self.init_sd_runner,
            name="init stable diffusion runner"
        )
        sd_runner_thread.join()

    def __init__(self, *args, **kwargs):
        """
        Initialize the worker
        """
        self.max_client_connections = kwargs.get("max_client_connections", 1)
        self.port = kwargs.get("port", settings.DEFAULT_PORT)
        self.host = kwargs.get("host", settings.DEFAULT_HOST)
        self.safety_model = kwargs.get("safety_model")
        self.model_name = kwargs.get("model_name")
        self.model_version = kwargs.get("model_version")
        self.safety_feature_extractor = kwargs.get("safety_feature_extractor")
        self.safety_model_path = kwargs.get("safety_model_path"),
        self.safety_feature_extractor_path = kwargs.get("safety_feature_extractor_path")

        self.sdrunner = SDRunner()

        self.do_start()
        super().__init__(*args, **kwargs)


# same class as StableDiffusionRequestQueueWorker but faster by not using a queue
# and not using a thread to send responses to the client
class FastStableDiffusionRequestQueueWorker(StableDiffusionRequestQueueWorker):
    def response_queue_worker(self):
        """
        Wait for responses from the stable diffusion runner and send
        them to the client
        """

    def handle_image(self, response, options):
        if response is not None and response != b'':
          # logger.info("prepping image")
            img = response #self.prep_image(response, options)
            try:
                bytes_sent = 0
                #expected_byte_size = settings.BYTE_SIZES[self.image_size]
                # actual_size = len(img)
                #
                # if actual_size < expected_byte_size:
                #     print("sending image of size {}".format(actual_size))
                #     # pad the image img_bytes with zeros
                #     img = img + b'\x00' * (expected_byte_size - actual_size)

                # send message in chunks
                chunk_size = 1024
                for i in range(0, len(img), chunk_size):
                    bytes_sent += self.send_msg(img[i:i + chunk_size])
                    time.sleep(0.001)
                time.sleep(0.1)

                logger.info(f"sent {bytes_sent} bytes")
                # send a message of all zeroes of expected_byte_size length
                # to indicate that the image is being sent
                bytes_sent = self.send_msg(b'\x00')
                logger.info(f"sent {bytes_sent} bytes")
                if self.soc_connection:
                    #self.soc_connection.settimeout(1)
                    pass
                else:
                    raise FailedToSendError()
            except FailedToSendError as ect:
                logger.error("failed to send message")
                # cancel the current run
                self.sdrunner.cancel()
                logger.error(ect)
                self.reset_connection()

    def __init__(self, *args, **kwargs):
        """
        Initialize the worker
        """
        super().__init__(*args, **kwargs)


if __name__ == '__main__':
    # app = Server(__name__)
    # app.run(debug=True)
    port = 5000
    host = "http://localhost"
    model_name = "runwayml/stable-diffusion-v1-5"
    app = FastStableDiffusionRequestQueueWorker(
        port=port,
        host=host,
        model_name=model_name
    )
