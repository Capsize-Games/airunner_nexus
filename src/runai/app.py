import json

from runai.agent import Agent
from runai.llm_handler import LLMHandler
from runai.llm_request import LLMRequest
from runai.logger import logger
from runai.socket_server import Server


class App:
    def __init__(self):
        self.llm_handler = None
        self.server = Server()
        self.llm_handler = LLMHandler()

    @staticmethod
    def parse_request_data(incoming_data: bytes) -> dict:
        """
        Parse incoming bytes from the client.
        :param incoming_data: bytes - incoming data from the client
        """
        try:
            data = incoming_data.decode("ascii")
        except UnicodeDecodeError as err:
            logger.error(f"something went wrong with a request from the client")
            logger.error(f"UnicodeDecodeError: {err}")
            return {}

        try:
            data = json.loads(data)
        except json.decoder.JSONDecodeError:
            logger.error(f"Improperly formatted request from client")
            return {}

        data["listener"] = Agent(**data["listener"])
        data["speaker"] = Agent(**data["speaker"])

        return data

    def query_llm(self, data: dict):
        llm_request = LLMRequest(**data)
        for text in self.llm_handler.query_model(llm_request):
            self.server.send_message(text)
        self.server.send_end_message()


if __name__ == '__main__':
    app = App()
