import json
import socket


class SocketClient:
    def __init__(self, host, port):
        self.packet_size = 1024
        self.host = host
        self.port = port
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def connect(self):
        self.client_socket.connect((self.host, self.port))

    def send_message(self, message):
        message = message.encode('utf-8')  # encode the message as UTF-8
        while message:
            packet = message[:self.packet_size]
            message = message[self.packet_size:]
            if len(packet) < self.packet_size:
                packet += b'\x00' * (self.packet_size - len(packet))  # pad the message with null bytes
            self.client_socket.sendall(packet)
        self.send_end_message()

    def send_end_message(self):
        # send a message of all zeroes of expected_byte_size length
        # to indicate that the image is being sent
        self.client_socket.sendall(b'\x00' * self.packet_size)

    def receive_message(self):
        while True:
            packet = self.client_socket.recv(self.packet_size)
            if packet == b'\x00' * self.packet_size:  # end message received
                break
            yield packet.decode('utf-8')  # decode the packet as UTF-8

    def close_connection(self):
        self.client_socket.close()
