import os
import torch
import threading
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.generation.streamers import TextIteratorStreamer
from airunner_nexus import settings
from airunner_nexus.llm.external_condition_stopping_criteria import ExternalConditionStoppingCriteria
from airunner_nexus.settings import MODEL_BASE_PATH, MODELS

class LLMHandler:
    def __init__(self, model_name: str = settings.DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self.model_path = os.path.join(os.path.expanduser(MODEL_BASE_PATH), MODELS[self.model_name]["path"])
        self.model = self.load_model()
        self.tokenizer = self.load_tokenizer()
        self.streamer = self.load_streamer()
        self.generate_thread = threading.Thread(target=self.generate)
        self.generate_data = None
        self._do_interrupt_process = False

    @property
    def quantized_model_path(self) -> str:
        return self.model_path + "_quantized"

    @property
    def device(self) -> str:
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    def resume(self):
        self._do_interrupt_process = False
        self.streamer = self.load_streamer()

    def interrupt(self):
        self._do_interrupt_process = True

    def load_model(self):
        model_path = self.quantized_model_path if os.path.exists(self.quantized_model_path) else self.model_path
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
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
        if model_path != self.quantized_model_path:
            model.save_pretrained(self.quantized_model_path)
        return model

    def load_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_path)

    def load_streamer(self):
        return TextIteratorStreamer(self.tokenizer)

    def query_model(self, data: dict):
        self.resume()
        conversation = data.get("conversation", [
            {"role": "system", "content": data.get("instructions", "")},
            {"role": "user", "content": data.get("prompt", "")},
        ])
        rendered_template = self.rendered_template(conversation)
        model_inputs = self.tokenizer(rendered_template, return_tensors="pt").to(self.device)
        stopping_criteria = ExternalConditionStoppingCriteria(self._do_interrupt_process)
        self.generate_data = dict(
            model_inputs,
            max_new_tokens=data.get("max_new_tokens", 1000),
            min_length=data.get("min_length", 0),
            do_sample=data.get("do_sample", True),
            early_stopping=data.get("early_stopping", True),
            num_beams=data.get("num_beams", 1),
            temperature=data.get("temperature", 0.9),
            top_p=data.get("top_p", 0.9),
            top_k=data.get("top_k", 50),
            repetition_penalty=data.get("repetition_penalty", 1.0),
            num_return_sequences=data.get("num_return_sequences", 1),
            decoder_start_token_id=data.get("decoder_start_token_id", None),
            use_cache=data.get("use_cache", True),
            length_penalty=data.get("length_penalty", 1.0),
            stopping_criteria=[stopping_criteria],
            streamer=self.streamer
        )

        if self.generate_thread.is_alive():
            self.generate_thread.join()

        self.generate_thread = threading.Thread(target=self.generate, args=(self.generate_data,))
        self.generate_thread.start()

        rendered_template = self.update_rendered_template(rendered_template)
        streamed_template = ""
        replaced = False

        for new_text in self.streamer:
            if not replaced:
                replaced, streamed_template = self.update_streamed_template(rendered_template, streamed_template, new_text)
            if replaced:
                yield self.strip_tags(new_text)

    def generate(self, data):
        self.model.generate(**data)

    def rendered_template(self, conversation: list) -> str:
        chat_template = MODELS[self.model_name]["chat_template"]
        return self.tokenizer.apply_chat_template(chat_template=chat_template, conversation=conversation, tokenize=False)

    @staticmethod
    def update_streamed_template(rendered_template: str, streamed_template: str, new_text: str) -> tuple:
        streamed_template += new_text.replace("</s>", "")
        replaced = rendered_template in streamed_template
        streamed_template = streamed_template.replace(rendered_template, "")
        return replaced, streamed_template

    @staticmethod
    def update_rendered_template(rendered_template: str) -> str:
        return "<s>" + rendered_template.replace("</s>", "").replace("<s>[INST] <<SYS>>", "<s>[INST]  <<SYS>>").replace("<</SYS>>[/INST][INST]", "<</SYS>>[/INST][INST] ")

    @staticmethod
    def strip_tags(template: str) -> str:
        return template.replace("[/INST]", "").replace("</s>", "").replace("<</SYS>>", "")
