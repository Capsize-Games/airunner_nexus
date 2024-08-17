from enum import Enum


class HandlerType(Enum):
    TRANSFORMER = 100
    DIFFUSER = 200


class LLMActionType(Enum):
    """
    The following action types are used by the LLM to process various user
    requests. The default action type is "Chat". This is used when the user
    wants to interact with a chatbot. When this is combined with the
    use_tool_flter flag, the LLM will attempt to determine which action to take
    based on the user's words.
    """
    DO_NOT_RESPOND = "DO NOT RESPOND: Use this option when the user has asked you to stop responding or if the text does not require a response."
    CHAT = "RESPOND TO THE USER: Respond to the user's message."
    GENERATE_IMAGE = "GENERATE IMAGE: Generate an image based on the text."
    ANALYZE_VISION_HISTORY = "ANALYZE VISION HISTORY: Analyze the vision history."
    APPLICATION_COMMAND = "APPLICATION COMMAND: Execute an application command."
    UPDATE_MOOD = "UPDATE MOOD: {{ username }} has made you feel a certain way. Respond with an emotion or feeling so that you can update your current mood."
    QUIT_APPLICATION = "QUIT: Quit or close the application."
    TOGGLE_FULLSCREEN = "FULL SCREEN: Make the application full screen."
    TOGGLE_TTS = "TOGGLE TTS: Toggle text-to-speech on or off."
    PERFORM_RAG_SEARCH = "SEARCH: Perform a search for information related to the user's query or context within the conversation."


class TTSModel(Enum):
    ESPEAK = "espeak"
    SPEECHT5 = "speecht5"
    BARK = "bark"


class FilterType(Enum):
    PIXEL_ART = "pixelart"


class AgentState(enum.Enum):
    SEARCH = enum.auto()
    CHAT = enum.auto()
