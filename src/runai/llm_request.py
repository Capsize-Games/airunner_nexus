from typing import Optional, List

from runai.agent import Agent
from runai.enums import LLMQueryType


class LLMRequest:
    def __init__(
        self,
        history: List[dict] = None,
        listener: Agent = None,
        speaker: Agent = None,
        use_usernames: bool = False,
        prompt_prefix: str = "",
        instructions: str = "",
        prompt: str = "",
        max_new_tokens: int = 1000,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 0.9,
        min_length: int = 0,
        do_sample: bool = True,
        num_beams: int = 1,
        repetition_penalty: float = 1.0,
        num_return_sequences: int = 1,
        decoder_start_token_id: Optional[int] = None,
        early_stopping: bool = True,
        use_cache: bool = True,
        length_penalty: float = 1.0,
        llm_query_type: Optional[LLMQueryType] = None
    ):
        self.history = history if history else []
        self.listener = listener
        self.speaker = speaker
        self.use_usernames = use_usernames
        self.prompt_prefix = prompt_prefix
        self._instructions = instructions
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.llm_query_type = llm_query_type
        self.min_length = min_length
        self.do_sample = do_sample
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.repetition_penalty = repetition_penalty
        self.num_return_sequences = num_return_sequences
        self.decoder_start_token_id = decoder_start_token_id
        self.use_cache = use_cache
        self.length_penalty = length_penalty

    @property
    def instructions(self):
        instructions = self._instructions
        if len(self.history):
            instructions += "\nThe conversation so far:\n"
            for turn in self.history:
                instructions += f"{turn['name']}: {turn['message']}\n"
        return instructions

    @property
    def conversation(self):
        return [
            {"role": "system", "content": self._instructions},
            {"role": "user", "content": self.prompt}
        ]

    def to_dict(self):
        return {
            "history": self.history,
            "listener": self.listener.to_dict() if self.listener else None,
            "speaker": self.speaker.to_dict() if self.speaker else None,
            "use_usernames": self.use_usernames,
            "prompt_prefix": self.prompt_prefix,
            "instructions": self.instructions,
            "prompt": self.prompt,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "llm_query_type": self.llm_query_type,
            "min_length": self.min_length,
            "do_sample": self.do_sample,
            "early_stopping": self.early_stopping,
            "num_beams": self.num_beams,
            "repetition_penalty": self.repetition_penalty,
            "num_return_sequences": self.num_return_sequences,
            "decoder_start_token_id": self.decoder_start_token_id,
            "use_cache": self.use_cache,
            "length_penalty": self.length_penalty
        }
