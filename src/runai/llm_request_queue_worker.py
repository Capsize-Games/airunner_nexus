import json
import settings
from llm_handler import LLMHandler
from logger import logger
from simple_enqueue_socket_server import SimpleEnqueueSocketServer


class LLMRequestQueueWorker(SimpleEnqueueSocketServer):
    """
    A socket server that listens for requests and enqueues them to a queue
    """
    llm_runner = None
    llm_handler = None

    def callback(self, data):
        """
        Handle a stable diffusion request message
        :return: None
        """
        try:
            # convert ascii to json
            data = data.decode("ascii")
        except UnicodeDecodeError as err:
            logger.error(f"something went wrong with a request from the client")
            logger.error(f"UnicodeDecodeError: {err}")
            return
        except json.decoder.JSONDecodeError as err:
            logger.error(f"Improperly formatted request from client")
            return

        try:
            data = json.loads(data)
        except json.decoder.JSONDecodeError:
            logger.error(f"Improperly formatted request from client")
            print(data)
            return

        for text in self.llm_handler.query_model(**data):
            self.send_message(text)
        self.send_end_message()

    def stop(self):
        super().stop()
        self.quit_event.set()

    def do_start(self):
        # create a stable diffusion runner service
        self.start_thread(
            target=self.response_queue_worker,
            name="response queue worker"
        )
        sd_runner_thread = self.start_thread(
            target=self.init_llm,
            name="init stable diffusion runner"
        )
        sd_runner_thread.join()

    def init_llm(self):
        """
        Initialize the stable diffusion runner
        return: None
        """
        logger.info("Starting Stable Diffusion Runner")
        self.llm_handler = LLMHandler()

    def response_queue_worker(self):
        pass

    def __init__(self, *args, **kwargs):
        """
        Initialize the worker
        """
        # self.db = DBConnection('database.db', b'key')
        #self.db.insert_txt2img_sample(1, 2, b'sample', 42, 100, 100, 10, 0.5, 'model', 'scheduler')

        self.model_base_path = kwargs.pop("model_base_path", None)
        self.max_client_connections = kwargs.get("max_client_connections", 1)
        self.port = kwargs.get("port", settings.DEFAULT_PORT)
        self.host = kwargs.get("host", settings.DEFAULT_HOST)
        self.do_timeout = kwargs.get("do_timeout", False)
        self.safety_model = kwargs.get("safety_model")
        self.model_version = kwargs.get("model_version")
        self.safety_feature_extractor = kwargs.get("safety_feature_extractor")
        self.safety_model_path = kwargs.get("safety_model_path"),
        self.safety_feature_extractor_path = kwargs.get("safety_feature_extractor_path")
        self.packet_size = kwargs.get("packet_size", settings.PACKET_SIZE)
        self.do_start()
        super().__init__(*args, **kwargs)
