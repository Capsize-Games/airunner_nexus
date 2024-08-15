import threading

from llama_index.core import SimpleDirectoryReader, ServiceContext, PromptHelper, \
    SimpleKeywordTableIndex
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.indices.keyword_table import KeywordTableSimpleRetriever
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core.readers.json import JSONReader

from runai.llm.enums import AgentState
from runai.llm.external_condition_stopping_criteria import ExternalConditionStoppingCriteria


class RagMixin:
    def __init__(self):
        self.embed_model = None
        self.json_reader = None
        self.file_extractor = None
        self.documents = None
        self.text_splitter = None
        self.prompt_helper = None
        self.service_context = None
        self.index = None
        self.retriever = None
        self.chat_engine = None
        self.__query_instruction = "Search through all available texts and provide a brief summary of the key points which are relevant to the query."
        self.__text_instruction = "Summarize and provide a brief explanation of the text. Stay concise and to the point."
        self.agent_state = AgentState.SEARCH

        self.load_rag_model()
        self.load_readers()
        self.load_file_extractor()
        self.load_documents()
        self.load_text_splitter()
        self.load_prompt_helper()
        self.load_service_context()
        self.load_document_index()
        self.load_retriever()
        self.load_context_chat_engine()

    @property
    def text_instruction(self):
        if self.agent_state == AgentState.SEARCH:
            return self.__text_instruction
        elif self.agent_state == AgentState.CHAT:
            return "Use the text to respond to the user"

    @property
    def query_instruction(self):
        if self.agent_state == AgentState.SEARCH:
            return self.__query_instruction
        elif self.agent_state == AgentState.CHAT:
            return "Search through the chat history for anything relevant to the query."
        else:
            return ""

    def load_rag_model(self):
        self.embed_model = HuggingFaceEmbedding(
            model_name=self.model_path,
            query_instruction=self.query_instruction,
            text_instruction=self.text_instruction,
            trust_remote_code=False,
        )

    def load_readers(self):
        self.json_reader = JSONReader()

    def load_file_extractor(self):
        self.file_extractor = {
            ".md": self.markdown_reader,
        }

    def load_documents(self):
        try:
            self.documents = SimpleDirectoryReader(
                input_files=self.target_files,
                file_extractor=self.file_extractor,
                exclude_hidden=False
            ).load_data()
        except ValueError as e:
            print(f"Error loading documents: {str(e)}")

    def load_text_splitter(self):
        self.text_splitter = SentenceSplitter(
            chunk_size=256,
            chunk_overlap=20
        )

    def load_prompt_helper(self):
        self.prompt_helper = PromptHelper(
            context_window=4096,
            num_output=1024,
            chunk_overlap_ratio=0.1,
            chunk_size_limit=None,
        )

    def load_service_context(self):
        try:
            # Update service context to use the newly created chat engine
            self.service_context = ServiceContext.from_defaults(
                llm=self.model,
                embed_model=Settings.embed_model,
                # chat_engine=self.chat_engine,  # Include the chat engine in the service context
                text_splitter=self.text_splitter,
                prompt_helper=self.prompt_helper,
            )
        except Exception as e:
            print(f"Error loading service context with chat engine: {str(e)}")

    def load_document_index(self):
        try:
            self.index = SimpleKeywordTableIndex.from_documents(
                self.documents,
                service_context=self.service_context,
            )
        except TypeError as e:
            print(f"Error loading index: {str(e)}")

    def load_retriever(self):
        try:
            self.retriever = KeywordTableSimpleRetriever(
                index=self.index,
            )
        except Exception as e:
            print(f"Error setting up the retriever: {str(e)}")

    def load_context_chat_engine(self):
        context_retriever = self.retriever  # Your method to retrieve context

        try:
            self.chat_engine = ContextChatEngine.from_defaults(
                retriever=context_retriever,
                service_context=self.service_context,
                chat_history=self.history,
                memory=None,  # Define or use an existing memory buffer if needed
                system_prompt="Search the full text and find all relevant information related to the query.",
                node_postprocessors=[],  # Add postprocessors if utilized in your setup
                llm=self.model,  # Use the existing LLM setup
            )
        except Exception as e:
            print(f"Error loading chat engine: {str(e)}")

    def rag_query_model(
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
        history_prompt=""
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
        print("*"*1000)
        print(rendered_template)
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
