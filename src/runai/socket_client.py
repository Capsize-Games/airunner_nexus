import json
import socket
import time


class SocketClient:
    def __init__(self, host, port, packet_size, retry_delay=2):
        self.packet_size = packet_size
        self.host = host
        self.port = port
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.retry_delay = retry_delay

    def connect(self):
        while True:
            try:
                self.client_socket.connect((self.host, self.port))
                print("Connected to server.")
                return
            except ConnectionRefusedError:
                print(f"Connection refused. Retrying in {self.retry_delay} seconds...")
                time.sleep(self.retry_delay)

    def send_message(self, message):
        message = message.encode('utf-8')  # encode the message as UTF-8
        while message:
            packet = message[:self.packet_size]
            message = message[self.packet_size:]
            if len(packet) < self.packet_size:
                packet += b'\x00' * (self.packet_size - len(packet))  # pad the message with null bytes
            try:
                self.client_socket.sendall(packet)
            except BrokenPipeError:
                print("Connection lost. Make sure the server is running.")
                break
        self.send_end_message()

    def send_end_message(self):
        # send a message of all zeroes of expected_byte_size length
        # to indicate that the image is being sent
        try:
            self.client_socket.sendall(b'\x00' * self.packet_size)
        except BrokenPipeError:
            print("Connection lost. Make sure the server is running.")

    def receive_message(self):
        while True:
            try:
                packet = self.client_socket.recv(self.packet_size)
            except OSError:
                print("Connection lost. Make sure the server is running.")
                break
            if packet == b'\x00' * self.packet_size:  # end message received
                break
            yield packet.decode('utf-8')  # decode the packet as UTF-8

    def close_connection(self):
        self.client_socket.close()
