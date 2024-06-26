import os
import torch
import threading

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.generation.streamers import TextIteratorStreamer

from runai.external_condition_stopping_criteria import ExternalConditionStoppingCriteria


class LLMHandler:
    def __init__(self):
        self._model_path = os.path.expanduser("~/.airunner/text/models/causallm")
        self._model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        self.rendered_template = None
        self.model = self.load_model()
        self.tokenizer = self.load_tokenizer()
        self.streamer = self.load_streamer()
        self.generate_thread = threading.Thread(target=self.generate)
        self.generate_data = None

    @property
    def model_path(self):
        return os.path.join(self._model_path, self._model_name)

    @property
    def device(self):
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    def do_interrupt_process(self):
        return False  # TODO

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
        conversation=None,
        max_new_tokens=1000,
        min_length=1,
        do_sample=True,
        early_stopping=True,
        num_beams=1,
        temperature=0.1,
        top_p=0.1,
        top_k=5,
        repetition_penalty=1.0,
        num_return_sequences=1,
        decoder_start_token_id=None,
        use_cache=True,
        length_penalty=1.5,
    ):
        chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "{{ '[INST] <<SYS>>' + message['content'] + ' <</SYS>>[/INST]' }}"
            "{% elif message['role'] == 'user' %}"
            "{{ '[INST]' + message['content'] + ' [/INST]' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ message['content'] + eos_token + ' ' }}"
            "{% endif %}"
            "{% endfor %}"
        )
        rendered_template = self.tokenizer.apply_chat_template(
            chat_template=chat_template,
            conversation=conversation if conversation is not None else [],
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
            max_new_tokens=max_new_tokens,
            min_length=min_length,
            do_sample=do_sample,
            early_stopping=early_stopping,
            num_beams=num_beams,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences,
            decoder_start_token_id=decoder_start_token_id,
            use_cache=use_cache,
            length_penalty=length_penalty,
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
        strip_template = "<s> " + rendered_template
        strip_template = strip_template.replace(" [INST]", "  [INST]")
        strip_template = strip_template.replace("<s>  [INST] <<SYS>>", "<s> [INST] <<SYS>>")

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
