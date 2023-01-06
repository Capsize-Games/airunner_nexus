import json
import signal
import socket
import threading
import time
import queue
import settings
import messagecodes as codes
from logger import logger
from exceptions import FailedToSendError, NoConnectionToClientError
from socket_connection import SocketConnection


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
            self.queue.put(b"quit")
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

    @property
    def signal_byte_size(self):
        return self.packet_size

    def is_expected_message(self, packet, byte):
        return packet == byte * self.signal_byte_size

    def is_quit_message(self, packet):
        return self.is_expected_message(packet, b'x')

    def is_cancel_message(self, packet):
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

    def get_packet(self):
        packet = self.soc_connection.recv(self.signal_byte_size)
        return packet

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
                    if total_timeouts >= 3 and self.do_timeout:
                        self.quit_event.set()
                        break
                except Exception as exc:  # pylint: disable=broad-except
                    logger.error(exc)

            if current_state is codes.AWAITING_MESSAGE:
                msg = None
                try:
                    packets = []
                    bytes_recd = 0
                    while True:
                        packet = self.get_packet()
                        # if packet size is 0 then it tells us the
                        # client is done sending the message
                        if packet == b'\x00' * self.packet_size:
                            break

                        # strip the packet of any null bytes
                        packet = packet.strip(b'\x00')

                        if packet == b'':
                            raise RuntimeError("socket connection broken")

                        if self.is_quit_message(packet):
                            self.handle_quit_message()
                            break

                        if self.is_cancel_message(packet):
                            self.handle_cancel_message()
                            break

                        # switch_model = self.is_model_switch_message(packet)
                        # if switch_model:
                        #     self.switch_model(switch_model)
                        #     break

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
        self.queue = None
        self.do_timeout = kwargs.get("do_timeout", False)
        if not self.queue:
            self.queue = queue.SimpleQueue()
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