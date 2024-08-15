import json
import re
import socket
import time
from datetime import datetime

from runai.llm.agent import Agent
from runai.settings import DEFAULT_HOST, DEFAULT_PORT, PACKET_SIZE, USER_NAME, BOT_NAME, LLM_INSTRUCTIONS


class Client:
    def __init__(
        self,
        host=DEFAULT_HOST,
        port=DEFAULT_PORT,
        packet_size=PACKET_SIZE,
        retry_delay=2,
        user_name=USER_NAME,
        bot_name=BOT_NAME
    ):
        self.host = host
        self.port = port
        self.packet_size = packet_size
        self.retry_delay = retry_delay
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.bot_agent = Agent(name=bot_name)
        self.user_agent = Agent(name=user_name)

        self.history = []

        self.connect()

    @property
    def dialogue_instructions(self):
        return LLM_INSTRUCTIONS["dialogue_instructions"].format(
            dialogue_rules=self.dialogue_rules,
            mood_stats=self.mood_stats,
            contextual_information=self.contextual_information
        )

    @property
    def contextual_information(self):
        return LLM_INSTRUCTIONS["contextual_information"].format(
            date_time=datetime.now().strftime("%m/%d/%Y, %I:%M:%S %p"),
            weather="sunny"
        )

    @property
    def update_mood_instructions(self):
        return LLM_INSTRUCTIONS["update_mood_instructions"].format(
            speaker_name=self.bot_agent.name,
            python_rules=self.python_rules
        )

    @property
    def mood_stats(self):
        stats = ", ".join([f"{k}: {v}" for k, v in self.bot_agent.mood_stats.items()])
        return (
            f"{self.bot_agent.name}'s mood stats:\n"
            f"{stats}\n"
        )

    @property
    def dialogue_rules(self):
        return LLM_INSTRUCTIONS["dialogue_rules"].format(
            speaker_name=self.bot_agent.name,
            listener_name=self.user_agent.name
        )

    @property
    def json_rules(self):
        return LLM_INSTRUCTIONS["json_rules"]

    @property
    def python_rules(self):
        return LLM_INSTRUCTIONS["python_rules"]

    def do_greeting(self):
        return self.do_query(
            LLM_INSTRUCTIONS["greeting_prompt"].format(speaker_name=self.bot_agent.name),
            self.dialogue_instructions
        )

    def do_response(self):
        return self.do_query(
            LLM_INSTRUCTIONS["response_prompt"].format(speaker_name=self.bot_agent.name),
            self.dialogue_instructions
        )

    def update_mood(self, agent: Agent):
        stats = ", ".join([f'"{k}": {v}' for k, v in agent.mood_stats.items()])
        res = self.do_query(
            LLM_INSTRUCTIONS["update_mood_prompt"].format(
                agent_name=agent.name,
                stats=stats
            ),
            self.update_mood_instructions
        )
        python_code_match = self.find_python(res)
        if python_code_match:
            python_code = python_code_match.group(1)
            exec(python_code, {}, {"agent": agent})
        return agent

    @staticmethod
    def find_python(res: str):
        return Client.find_code_block("python", res)

    @staticmethod
    def find_json(res: str):
        return Client.find_code_block("json", res)

    @staticmethod
    def find_code_block(language: str, res: str) -> re.Match:
        return re.search(r'```' + language + 'r\n(.*?)\n```', res, re.DOTALL)

    def do_prompt(self, user_prompt, update_speaker_mood=False):
        self.update_history(self.user_agent.name, user_prompt)

        if update_speaker_mood:
            self.bot_agent = self.update_mood(self.bot_agent)

        for res in self.do_response():
            yield res

        return ""

    def update_history(self, name: str, message: str):
        self.history.append({
            "name": name,
            "message": message
        })

    def do_query(self, user_prompt, instructions):
        self.send_message(json.dumps({
            "history": self.history,
            "listener": self.user_agent.to_dict() if self.user_agent else None,
            "speaker": self.bot_agent.to_dict() if self.bot_agent else None,
            "use_usernames": True,
            "prompt_prefix": "",
            "instructions": instructions,
            "prompt": user_prompt,
            "max_new_tokens": 1000,
            "temperature": 0.9,
            "top_k": 50,
            "top_p": 0.9,
            "query_type": "llm",
            "min_length": 0,
            "do_sample": True,
            "early_stopping": True,
            "num_beams": 1,
            "repetition_penalty": 1.0,
            "num_return_sequences": 1,
            "decoder_start_token_id": None,
            "use_cache": True,
            "length_penalty": 1.0
        }))

        server_response = ""
        for res in self.receive_message():
            res.replace(f"{self.bot_agent.name}: ", "")
            server_response += res.replace('\x00', '')
            yield server_response
        return server_response.replace('\x00', '')

    def _handle_prompt(self, prompt: str):
        response = ""
        for txt in self.do_prompt(prompt):
            response = str(txt)
            yield response

        self.history.append({
            "name": self.bot_agent.name,
            "message": response
        })

    def run(self):
        while True:
            prompt = input("Enter a prompt: ")
            for res in self._handle_prompt(prompt):
                self.handle_res(res)

    def handle_res(self, res):
        print(res)

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


if __name__ == "__main__":
    client = Client()
    client.run()
