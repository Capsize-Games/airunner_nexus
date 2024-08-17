import json
import re
import signal
import socket
import threading
import time
import queue
from typing import Optional

from airunner_nexus import settings
from airunner_nexus.llm.llm_handler import LLMHandler
from airunner_nexus.logger import logger
from airunner_nexus.exceptions import FailedToSendError, NoConnectionToClientError
import airunner_nexus.messagecodes as codes


class Server:
    def __init__(self, *args, **kwargs):
        self.max_clients = kwargs.get("max_clients", settings.MAX_CLIENTS)
        self.port = kwargs.get("port", settings.DEFAULT_PORT)
        self.host = kwargs.get("host", settings.DEFAULT_HOST)
        self.packet_size = kwargs.get("packet_size", settings.PACKET_SIZE)
        self.max_client_connections = kwargs.get("max_client_connections", 1)
        self.model_base_path = kwargs.get("model_base_path", ".")
        self.do_timeout = kwargs.get("do_timeout", False)

        self.soc = None
        self.soc_connection = None
        self.soc_addr = None
        self.threads = []
        self.queue = queue.SimpleQueue()
        self.quit_event = threading.Event()
        self.has_connection = False
        self.llm_handler = LLMHandler()
        self.message = None

        self.initialize_socket()
        self.start()
        signal.signal(signal.SIGINT, self.quit_event.set)  # handle ctrl+c
        self.start_thread(target=self.worker, name="socket server worker")
        self.start_thread(target=self.watch_connection, name="watch connection")

    @property
    def message(self) -> str:
        """Does nothing. Only used for the setter."""
        return ""

    @message.setter
    def message(self, msg: bytes):
        """Place incoming messages onto the queue."""
        self.queue.put(msg)

    @property
    def signal_byte_size(self) -> int:
        return self.packet_size

    @staticmethod
    def find_json(res: str) -> re.Match:
        return Server.find_code_block("json", res)

    @staticmethod
    def find_code_block(language: str, res: str) -> re.Match:
        return re.search(r'```' + language + '(.*?)```', res, re.DOTALL)

    @staticmethod
    def parse_request_data(incoming_data: bytes) -> dict:
        """Parse incoming bytes from the client."""
        try:
            data = incoming_data.decode("ascii")
        except UnicodeDecodeError as err:
            logger.error("something went wrong with a request from the client")
            logger.error(f"UnicodeDecodeError: {err}")
            return {}

        try:
            data = json.loads(data)
        except json.decoder.JSONDecodeError:
            logger.error("Improperly formatted request from client")
            return {}

        return data

    def worker(self):
        """Start a worker to handle request queue."""
        logger.info("Enqueue worker started")
        while not self.quit_event.is_set():
            if self.has_connection:
                try:
                    msg = self.queue.get(timeout=1)  # get a message from the queue
                    if msg is not None:
                        logger.info("Received message from queue")
                        try:
                            self.handle_message(msg)
                        except Exception as err:
                            logger.info(f"callback error: {err}")
                            raise err
                except queue.Empty:
                    pass
            if self.quit_event.is_set():
                break
            time.sleep(1)
        logger.info("SERVER WORKER: worker stopped")

    def start(self):
        """Starts a new thread with a connection to service."""
        self.start_thread(target=self.connect, name="Connection thread")

    def stop(self):
        """Disconnects from service and stops the thread."""
        self.disconnect()
        logger.info("Stopping connection thread...")
        for index, thread in enumerate(self.threads):
            total = len(self.threads)
            name = thread.name
            logger.info(f"{index + 1} of {total} Stopping thread {name}")
            try:
                thread.join()
            except RuntimeError:
                logger.info(f"Thread {thread.name} not running")
            logger.info(f"Stopped thread {thread.name}...")
        logger.info("All threads stopped")
        self.quit_event.set()

    def start_thread(self, target: Optional, daemon: bool = False, name: str = None) -> threading.Thread:
        """Start a thread and append it to the list of threads on this object."""
        thread = threading.Thread(target=target, daemon=daemon)
        if name:
            thread.name = name
        thread.start()
        self.threads.append(thread)
        return thread

    def connect(self):
        """Open a socket and handle connection."""
        self.open_socket()
        self.handle_open_socket()

    def disconnect(self):
        """Disconnect from socket."""
        if self.soc_connection:
            self.soc_connection.close()
        self.soc.close()
        self.soc_connection = None

    def reconnect(self):
        """Disconnects then reconnects to service. Does not stop the thread."""
        self.disconnect()
        self.connect()

    def initialize_socket(self):
        """Initialize a socket. Use timeout to prevent constant blocking."""
        self.soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.soc.settimeout(3)

    def reset_connection(self):
        """Reset connection to service."""
        self.disconnect()
        self.initialize_socket()
        self.has_connection = False
        self.open_socket()
        self.listen_to_socket()

    def handle_message(self, msg: bytes):
        """Override this method or pass it in as a parameter to handle messages."""
        data = self.parse_request_data(msg)
        self.query_llm(data)

    def open_socket(self):
        """Open a socket connection."""
        try:
            self.soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.soc.settimeout(1)
            self.soc.bind(("", 50006))
        except socket.error as err:
            logger.info(f"Failed to open a socket at {self.host}:{self.port}")
            logger.info(str(err))
        except Exception as _exc:
            logger.error(f"Failed to open a socket at {self.host}:{self.port}")
            logger.error(_exc)
        logger.info(f"Socket opened {self.soc}")

    def listen_to_socket(self):
        """Listen to socket for connections."""
        self.soc.listen(self.max_clients)
        logger.info(f"Listening for connections on {self.host}:{self.port}")

    def try_quit(self) -> bool:
        """Try to quit the thread."""
        if self.quit_event.is_set():
            self.queue.put(b"quit")
            if self.soc_connection:
                self.soc_connection.close()
                self.soc_connection = None
            if self.queue:
                self.queue.put("quit")
            return True
        return False

    def do_send(self, msg: bytes) -> int:
        """Send a message to the client."""
        size_in_bytes = len(msg)
        bytes_sent = 0
        if not self.soc_connection:
            logger.error("No connection to client")
        else:
            try:
                bytes_sent = self.soc_connection.send(msg)
            except Exception as e:
                logger.error("something went wrong")
                logger.error(e)
            if bytes_sent != size_in_bytes:
                logger.error("Failed to send all bytes")
        return bytes_sent

    def message_client(self, message: dict):
        """Convenience method to send a message to the client."""
        message = json.dumps(message).encode()
        self.send_message(message)
        self.send_end_message()

    def send_message(self, message: bytes):
        """Send a message to the client in byte packets."""
        packet_size = self.packet_size
        for i in range(0, len(message), packet_size):
            packet = message[i:i + packet_size]
            self.do_send(packet + b'\x00' * (packet_size - len(packet)))

    def send_end_message(self):
        """Send a message of all zeroes of expected_byte_size length to indicate that the image is being sent."""
        self.do_send(b'\x00' * self.packet_size)

    def send_msg(self, msg: Optional[bytes] = None) -> int:
        """Send a message to the client."""
        success = False
        bytes_sent = 0
        try:
            bytes_sent = self.do_send(msg)
            success = True
        except FailedToSendError:
            logger.error("failed to send connection to client")
        except NoConnectionToClientError:
            logger.error("Lost connection to client")
        if not success:
            logger.error("failed to send message, adding back to queue")
            self.message = msg
        return bytes_sent

    def is_expected_message(self, packet: bytes, byte: bytes) -> bool:
        return packet == byte * self.signal_byte_size

    def is_quit_message(self, packet: bytes) -> bool:
        return self.is_expected_message(packet, b'x')

    def is_cancel_message(self, packet: bytes) -> bool:
        return self.is_expected_message(packet, b'c')

    def handle_quit_message(self):
        logger.info("Quit")
        self.quit_event.set()
        self.message = None
        self.queue.put(b"quit")

    def handle_cancel_message(self):
        logger.info("Cancel image")
        self.message = None
        self.cancel()

    def handle_model_switch_message(self, model: str):
        pass

    def switch_model(self, model: str):
        logger.info("switch_model")
        self.message = json.dumps({
            "reqtype": "switch_model",
            "model": model
        }).encode()

    def get_packet(self) -> bytes:
        return self.soc_connection.recv(self.signal_byte_size)

    def handle_open_socket(self):
        """Listen for incoming connections."""
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
                    if total_timeouts >= 3 and self.do_timeout:
                        self.quit_event.set()
                        break
                except Exception as exc:
                    logger.error(exc)

            if current_state is codes.AWAITING_MESSAGE:
                msg = None
                try:
                    packets = []
                    while True:
                        packet = self.get_packet()
                        if packet == b'\x00' * self.packet_size:
                            break
                        packet = packet.strip(b'\x00')
                        if packet == b'':
                            raise RuntimeError("socket connection broken")
                        if self.is_quit_message(packet):
                            self.handle_quit_message()
                            break
                        if self.is_cancel_message(packet):
                            self.handle_cancel_message()
                            break
                        if packet != b'':
                            packets.append(packet)
                    msg = b''.join(packets)
                except socket.timeout:
                    pass
                except AttributeError:
                    logger.error("attribute error")
                except ConnectionResetError:
                    logger.error("connection reset error")
                    current_state = codes.AWAITING_CONNECTION
                except Exception as exc:
                    logger.error(exc)
                    current_state = codes.AWAITING_CONNECTION

                if msg is not None and msg != b'':
                    logger.info("message received")
                    self.message = msg  # push directly to queue
                    self.soc_connection.settimeout(None)
                    current_state = codes.AWAITING_MESSAGE
                else:
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
        self.stop()

    def cancel(self):
        pass

    def watch_connection(self):
        """Watch the connection and shutdown if the server is the connection is lost."""
        while not self.quit_event.is_set():
            if self.try_quit():
                logger.info("shutting down")
                break
            time.sleep(1)

    def query_llm(self, data: dict):
        do_json = True
        response = ""
        for text in self.llm_handler.query_model(data):
            response += text
            if not do_json:
                self.send_message(text)

        if do_json:
            response = response.strip().replace("\n", " ")
            found_json = Server.find_json(response)
            if found_json:
                response = found_json.group(1)
                self.send_message(response)
            else:
                self.send_message(response)

        self.send_end_message()


if __name__ == '__main__':
    server = Server()
