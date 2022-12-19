import base64
import json
import os
import signal
import socket
import threading
import time
from typing import Optional
import io
import numpy as np
import torch
import queue
from transformers import CLIPFeatureExtractor
import settings
import messagecodes as codes
from PIL import Image
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    EulerAncestralDiscreteScheduler,
    DDPMScheduler,
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler, StableDiffusionPipelineSafe
)
from pytorch_lightning import seed_everything
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from logger import logger
from exceptions import FailedToSendError, NoConnectionToClientError

LONG_MESSAGE_SIZE=5610434
RESPONSE_QUEUE = queue.SimpleQueue()


class SDRunner:
    _current_model = ""
    scheduler_name = "ddpm"
    do_nsfw_filter = True
    do_watermark = True
    initialized = False

    @property
    def current_model(self):
        return self._current_model

    @current_model.setter
    def current_model(self, model):
        if self._current_model != model:
            self._current_model = model
            if self.initialized:
                self.load_model()

    @property
    def model_path(self):
        model_path = self.current_model
        if self.current_model in [
            "stable-diffusion-v1-5",
            "stable-diffusion-inpainting",
            "stable-diffusion-2-1-base",
            "stable-diffusion-2-inpainting",
        ]:
            model_path = f"./models/{self.current_model}"
        return model_path

    def load_model(self):
        torch.cuda.empty_cache()
        # load StableDiffusionSafetyChecker with CLIPConfig
        self.safety_checker = StableDiffusionSafetyChecker(
            StableDiffusionSafetyChecker.config_class()
        )
        self.feature_extractor = CLIPFeatureExtractor()

        # check if self.current_model has ckpt extension
        # if self.current_model.endswith(".ckpt"):
        #     print("found checkpoint file")
        #     self.current_model = "/home/joe/Projects/ai/runai2/stablediffusion/stable-diffusion-v1-5"
        # self.current_model = "/home/joe/Projects/ai/runai2/models/stable-diffusion-v1-5"

        if self.do_nsfw_filter:
            self.txt2img = StableDiffusionPipelineSafe.from_pretrained(
                self.model_path,
                torch_dtype=torch.half,
                scheduler=self.scheduler,
                low_cpu_mem_usage=True,
                # safety_checker=self.safety_checker,
                # feature_extractor=self.feature_extractor,
                # revision="fp16"
            )
        else:
            self.txt2img = StableDiffusionPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.half,
                scheduler=self.scheduler,
                low_cpu_mem_usage=True,
                safety_checker=None,
            )
        self.txt2img.enable_xformers_memory_efficient_attention()
        self.txt2img.to("cuda")
        self.img2img = StableDiffusionImg2ImgPipeline(**self.txt2img.components)
        self.inpaint = StableDiffusionInpaintPipeline(**self.txt2img.components)

    schedulers = {
        "ddpm": DDPMScheduler,
        "ddim": DDIMScheduler,
        "plms": PNDMScheduler,
        "lms": LMSDiscreteScheduler,
        "euler_a": EulerAncestralDiscreteScheduler,
        "euler": EulerDiscreteScheduler,
        "dpm": DPMSolverMultistepScheduler,
    }

    registered_schedulers = {}

    @property
    def scheduler(self):
        if not self.model_path or self.model_path == "":
            raise Exception("Chicken / egg problem, model path not set")
        if self.scheduler_name in self.schedulers:
            if self.scheduler_name not in self.registered_schedulers:
                self.registered_schedulers[self.scheduler_name] = self.schedulers[self.scheduler_name].from_pretrained(
                    self.model_path,
                    subfolder="scheduler"
                )
            return self.registered_schedulers[self.scheduler_name]
        else:
            raise ValueError("Invalid scheduler name")

    def change_scheduler(self):
        if self.model_path and self.model_path != "":
            self.txt2img.scheduler = self.scheduler
            self.img2img.scheduler = self.scheduler
            self.inpaint.scheduler = self.scheduler


    def generator_sample(self, data, image_handler):
        self.image_handler = image_handler
        return self.generate(data)

    def convert(self, model):
        # get location of .ckpt file
        model_path = model.replace(".ckpt", "")
        model_name = model_path.split("/")[-1]

        required_files = [
            "feature_extractor/preprocessor_config.json",
            "safety_checker/config.json",
            "safety_checker/pytorch_model.bin",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "text_encoder/pytorch_model.bin",
            "tokenizer/merges.txt",
            "tokenizer/vocab.json",
            "tokenizer/tokenizer_config.json",
            "tokenizer/special_tokens_map.json",
            "unet/config.json",
            "unet/diffusion_pytorch_model.bin",
            "vae/config.json",
            "vae/diffusion_pytorch_model.bin",
            "model_index.json",
        ]

        missing_files = False
        for required_file in required_files:
            if not os.path.isfile(f"{model_path}/{required_file}"):
                logger.warning(f"missing file {model_path}/{required_file}")
                missing_files = True
                break

        if missing_files:
            dump_path = f"./models/stablediffusion/{model_name}"
            version = "v1-5"
            from scripts.convert import convert
            logger.info("Converting model")
            convert(
                extract_ema=True,
                checkpoint_path=model,
                dump_path=model_path,
                original_config_file=f"./models/stable-diffusion-{version}/v1-inference.yaml",
            )
            logger.info("ckpt converted to diffusers")
        return model_path

    def initialize(self):
        self.load_model()
        self.initialized = True

    def generate(self, data):
        options = data["options"]

        # Get the other values from options
        action = data.get("action", "txt2img")

        scheduler_name = options.get(f"{action}_scheduler", "ddpm")
        if self.scheduler_name != scheduler_name:
            self.scheduler_name = scheduler_name
            self.change_scheduler()
        #
        # # get model and switch to it
        model = options.get(f"{action}_model", self.current_model)

        print("MODEL REQUESTED: ", model)

        # if model is ckpt
        if model.endswith(".ckpt"):
            model = self.convert(model)

        if action in ["inpaint", "outpaint"]:
            if model in [
                "stable-diffusion-2-1-base",
                "stable-diffusion-2-base"
            ]:
                model = "stable-diffusion-2-inpainting"
            else:
                model = "stable-diffusion-inpainting"

        if model != self.current_model:
            self.current_model = model

        if not self.initialized:
            self.initialize()

        seed = int(options.get(f"{action}_seed", 42))
        guidance_scale = float(options.get(f"{action}_scale", 7.5))
        num_inference_steps = int(options.get(f"{action}_ddim_steps", 50))
        self.num_inference_steps = num_inference_steps
        self.strength = float(options.get(f"{action}_strength", 1.0))

        do_nsfw_filter = bool(options.get(f"do_nsfw_filter", False))
        do_watermark = bool(options.get(f"do_watermark", False))
        enable_community_models = bool(options.get(f"enable_community_models", False))
        if do_nsfw_filter != self.do_nsfw_filter:
            self.do_nsfw_filter = do_nsfw_filter
            self.load_model()
        if do_watermark != self.do_watermark:
            self.do_watermark = do_watermark
            self.load_model()
        prompt = options.get(f"{action}_prompt", "")
        negative_prompt = options.get(f"{action}_negative_prompt", "")
        C = int(options.get(f"{action}_C", 4))
        f = int(options.get(f"{action}_f", 8))
        batch_size = int(data.get(f"{action}_n_samples", 1))

        # sample the model
        with torch.no_grad() as _torch_nograd, \
            torch.cuda.amp.autocast() as _torch_autocast:
            try:
                # clear cuda cache
                for n in range(0, batch_size):
                    seed = seed + n
                    print("GETTING READY TO SEED WITH ", seed)
                    seed_everything(seed)
                    image = None
                    if action == "txt2img":
                        image = self.txt2img(
                            prompt,
                            negative_prompt=negative_prompt,
                            guidance_scale=guidance_scale,
                            num_inference_steps=num_inference_steps,
                            callback=self.callback
                        ).images[0]
                    elif action == "img2img":
                        bytes = base64.b64decode(data["options"]["pixels"])
                        image = Image.open(io.BytesIO(bytes))
                        image = self.img2img(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            image=image.convert("RGB"),
                            strength=self.strength,
                            guidance_scale=guidance_scale,
                            num_inference_steps=num_inference_steps,
                            callback=self.callback
                        ).images[0]
                        pass
                    elif action in ["inpaint", "outpaint"]:
                        bytes = base64.b64decode(data["options"]["pixels"])
                        mask_bytes = base64.b64decode(data["options"]["mask"])

                        image = Image.open(io.BytesIO(bytes))
                        mask = Image.open(io.BytesIO(mask_bytes))

                        # convert mask to 1 channel
                        # print mask shape
                        image = self.inpaint(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            image=image,
                            mask_image=mask,
                            guidance_scale=guidance_scale,
                            num_inference_steps=num_inference_steps,
                            callback=self.callback
                        ).images[0]
                        pass

                    # use pillow to convert the image to a byte array
                    if image:
                        img_byte_arr = io.BytesIO()
                        image.save(img_byte_arr, format='PNG')
                        img_byte_arr = img_byte_arr.getvalue()
                        #return flask.Response(img_byte_arr, mimetype='image/png')
                        self.image_handler(img_byte_arr, data)
            except TypeError as e:
                if action in ["inpaint", "outpaint"]:
                    print(f"ERROR IN {action}")
                print(e)
            # except Exception as e:
            #     print("Error during generation 1")
            #     print(e)
            #     #return flask.jsonify({"error": str(e)})

    def callback(self, step, time_step, latents):
        self.tqdm_callback(step, int(self.num_inference_steps * self.strength))

    def __init__(self, *args, **kwargs):
        self.tqdm_callback = kwargs.get("tqdm_callback", None)
        super().__init__(*args)


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
            name = thread.name
            logger.info(f"{index+1} of {total} Stopping thread {name}")
            try:
                thread.join()
            except RuntimeError:
                logger.info(f"Thread {thread.name} not running")
            logger.info(f"Stopped thread {thread.name}...")
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
        pass

    def worker(self):
        """
        Worker is started in a thread and waits for messages that are appended
        to the queue. When a message is received, it is passed to the callback
        method. The callback method should be overridden to handle the message.
        :return:
        """
        pass

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
                        if chunk == b'\x00' * 1024:
                            break

                        # strip the chunk of any null bytes
                        chunk = chunk.strip(b'\x00')

                        if chunk == b'':
                            raise RuntimeError("socket connection broken")

                        if self.is_quit_message(chunk):
                            self.handle_quit_message()
                            break

                        if self.is_cancel_message(chunk):
                            self.handle_cancel_message()
                            break

                        # switch_model = self.is_model_switch_message(chunk)
                        # if switch_model:
                        #     self.switch_model(switch_model)
                        #     break

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
            self.image_size = data.get("W", None)

            reqtype = data["action"]
            # for k,v in settings.ACTIONS.items():
            #     if k == actionID:
            #         reqtype = k
            #         break
        except UnicodeDecodeError as err:
            logger.error(f"something went wrong with a request from the client")
            logger.error(f"UnicodeDecodeError: {err}")
            return
        except json.decoder.JSONDecodeError as err:
            logger.error(f"Improperly formatted request from client")
            return
        data["reqtype"] = reqtype
        self.reqtype = reqtype
        if reqtype in ["txt2img", "img2img", "inpaint", "outpaint"]:
            self.sdrunner.generator_sample(data, self.handle_image)
            logger.info("Image sample complete")
        else:
            logger.error(f"NO IMAGE RESPONSE for reqtype {reqtype}")

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

    def tqdm_callback(self, step, total_steps):
        msg = {
            "action": codes.PROGRESS, 
            "step": step, 
            "total":total_steps,
            "reqtype": self.reqtype
        }
        msg = json.dumps(msg).encode()
        msg = msg + b'\x00' * (1024 - len(msg))
        self.do_send(msg)
        self.send_end_message()
        time.sleep(0.001)

    def send_image_chunk(self, image_chunk):
        """
        Send an image chunk to the client
        :param image_chunk: the image chunk to send
        :return: None
        """
        chunk = image_chunk + b'\x00' * (1024 - len(image_chunk))
        self.do_send(chunk)
        time.sleep(0.001)

    def send_end_message(self):
        # send a message of all zeroes of expected_byte_size length
        # to indicate that the image is being sent
        self.do_send(b'\x00' * 1024)
        time.sleep(0.001)

    def __init__(self, *args, **kwargs):
        """
        Initialize the worker
        """
        self.max_client_connections = kwargs.get("max_client_connections", 1)
        self.port = kwargs.get("port", settings.DEFAULT_PORT)
        self.host = kwargs.get("host", settings.DEFAULT_HOST)
        self.safety_model = kwargs.get("safety_model")
        self.model_version = kwargs.get("model_version")
        self.safety_feature_extractor = kwargs.get("safety_feature_extractor")
        self.safety_model_path = kwargs.get("safety_model_path"),
        self.safety_feature_extractor_path = kwargs.get("safety_feature_extractor_path")

        self.sdrunner = SDRunner(
            tqdm_callback=self.tqdm_callback
        )

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

    def handle_image(self, image, options):
        print("HANDLE IMAGE RESPONSE")
        # image is bytes and therefore not json serializable,
        # convert it to base64 first
        image = base64.b64encode(image).decode()
        response = {
            "image": image,
            "reqtype": options["reqtype"],
            "pos_x": options["options"]["pos_x"],
            "pos_y": options["options"]["pos_y"],
        }
        # encode response as a byte string
        response = json.dumps(response).encode()
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
                    chunk = img[i:i + chunk_size]
                    self.send_image_chunk(chunk)

                time.sleep(0.001)
                self.send_end_message()

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
    app = FastStableDiffusionRequestQueueWorker(
        port=port,
        host=host
    )
