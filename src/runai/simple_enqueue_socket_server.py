import time
import queue
from logger import logger
from socket_server import SocketServer


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
