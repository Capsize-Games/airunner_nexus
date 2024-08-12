import enum


class LLMQueryType(enum.Enum):
    DIALOGUE = enum.auto()


class AgentState(enum.Enum):
    SEARCH = enum.auto()
    CHAT = enum.auto()
