"""
Collection of exceptions
"""


class FailedToSendError(Exception):
    """Failed to send message to server"""
    message = "Failed to send message"


class NoConnectionToClientError(Exception):
    """No connection to client"""
    message = "No connection to client"