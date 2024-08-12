from typing import List, Optional

from runai.agent import Agent
from runai.enums import LLMQueryType


class LLMRequest:
    def __init__(
        self,
        conversation: Optional[List[dict]] = None,
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
        self.conversation = conversation
        self.listener = listener
        self.speaker = speaker
        self.use_usernames = use_usernames
        self.prompt_prefix = prompt_prefix
        self.instructions = instructions
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
