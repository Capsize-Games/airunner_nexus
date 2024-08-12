import os
import torch
import threading

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.generation.streamers import TextIteratorStreamer

from runai.external_condition_stopping_criteria import ExternalConditionStoppingCriteria
from runai.llm_request import LLMRequest
from runai.rag_mixin import RagMixin
from runai.settings import MODEL_BASE_PATH, MODELS


class LLMHandler(RagMixin):
    def __init__(self, model_name: str = ""):
        self._model_path = os.path.expanduser(MODEL_BASE_PATH)
        self.model_name = MODELS[model_name]["path"]

        # RagMixin.__init__(self)
        self.rendered_template = None
        self.model = self.load_model()
        self.tokenizer = self.load_tokenizer()
        self.streamer = self.load_streamer()
        self.generate_thread = threading.Thread(target=self.generate)
        self.generate_data = None
        self._do_interrupt_process = False

    def resume(self):
        self._do_interrupt_process = False

    def interrupt(self):
        self._do_interrupt_process = True

    @property
    def model_path(self):
        return os.path.join(self._model_path, self.model_name)

    @property
    def device(self):
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    def do_interrupt_process(self):
        return self._do_interrupt_process

    def load_model(self):
        return AutoModelForCausalLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            use_cache=True,
            trust_remote_code=False,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                load_in_8bit=False,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
            ),
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )

    def load_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_path)

    def load_streamer(self):
        return TextIteratorStreamer(self.tokenizer)

    def query_model(
        self,
        llm_request: LLMRequest
    ):
        chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "{{ '[INST] <<SYS>>' + message['content'] + ' <</SYS>>[/INST]' }}"
            "{% elif message['role'] == 'user' %}"
            "{{ '[INST]' + message['content'] + ' [/INST]' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ message['content'] + eosets_token + ' ' }}"
            "{% endif %}"
            "{% endfor %}"
        )

        rendered_template = self.tokenizer.apply_chat_template(
            chat_template=chat_template,
            conversation=llm_request.conversation,
            tokenize=False
        )
        self.rendered_template = rendered_template
        model_inputs = self.tokenizer(
            rendered_template,
            return_tensors="pt"
        ).to(self.device)
        stopping_criteria = ExternalConditionStoppingCriteria(
            self.do_interrupt_process
        )
        self.generate_data = dict(
            model_inputs,
            max_new_tokens=llm_request.max_new_tokens,
            min_length=llm_request.min_length,
            do_sample=llm_request.do_sample,
            early_stopping=llm_request.early_stopping,
            num_beams=llm_request.num_beams,
            temperature=llm_request.temperature,
            top_p=llm_request.top_p,
            top_k=llm_request.top_k,
            repetition_penalty=llm_request.repetition_penalty,
            num_return_sequences=llm_request.num_return_sequences,
            decoder_start_token_id=llm_request.decoder_start_token_id,
            use_cache=llm_request.use_cache,
            length_penalty=llm_request.length_penalty,
            stopping_criteria=[stopping_criteria],
            streamer=self.streamer
        )

        # If the thread is already running, wait for it to finish
        if self.generate_thread and self.generate_thread.is_alive():
            self.generate_thread.join()

        # Start the thread
        self.generate_thread = threading.Thread(
            target=self.generate,
            args=(self.generate_data,)
        )
        self.generate_thread.start()

        rendered_template = rendered_template.replace("</s>", "")
        strip_template = "<s>" + rendered_template
        # strip_template = strip_template.replace(" [INST]", "  [INST]")
        # strip_template = strip_template.replace("<s>  [INST] <<SYS>>", "<s>[INST]  <<SYS>>")


        strip_template = strip_template.replace("<s>[INST] <<SYS>>", "<s>[INST]  <<SYS>>")
        strip_template = strip_template.replace("<</SYS>>[/INST][INST]", "<</SYS>>[/INST][INST] ")

        streamed_template = ""
        replaced = False
        for new_text in self.streamer:
            streamed_template += new_text
            streamed_template = streamed_template.replace("</s>", "")
            if streamed_template.find(strip_template) != -1:
                replaced = True
            streamed_template = streamed_template.replace(strip_template, "")
            if replaced:
                parsed = new_text.replace("[/INST]", "")
                parsed = parsed.replace("</s>", "")
                parsed = parsed.replace("<</SYS>>", "")
                yield parsed

    def generate(self, data):
        self.model.generate(**data)
