import json
import io
import numpy as np
import settings
import messagecodes as codes
from PIL import Image
from runner import SDRunner
from logger import logger
from exceptions import FailedToSendError, NoConnectionToClientError
from simple_enqueue_socket_server import SimpleEnqueueSocketServer


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
            self.queue.put(img)

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
            res = self.queue.get()
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
            #time.sleep(0.1)
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
        #time.sleep(0.001)

    def send_image_chunk(self, image_chunk):
        """
        Send an image chunk to the client
        :param image_chunk: the image chunk to send
        :return: None
        """
        chunk = image_chunk + b'\x00' * (1024 - len(image_chunk))
        self.do_send(chunk)
        #time.sleep(0.001)

    def send_end_message(self):
        # send a message of all zeroes of expected_byte_size length
        # to indicate that the image is being sent
        self.do_send(b'\x00' * 1024)
        #time.sleep(0.001)

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