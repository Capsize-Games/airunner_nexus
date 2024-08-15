PACKET_SIZE = 1024
DEFAULT_PORT = 50006
DEFAULT_HOST = "0.0.0.0"
USER_NAME = "User"
BOT_NAME = "AI Bot"
MAX_CLIENTS = 1
DEBUG = True
DEFAULT_SERVER_TYPE = "LLM"
MODEL_BASE_PATH = "~/.airunner/text/models/causallm"
MODELS = {
    "mistral_instruct": {
        "path": "mistralai/Mistral-7B-Instruct-v0.3",
        "chat_template": (
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
    }
}
DEFAULT_MODEL_NAME = "mistral_instruct"
LLM_INSTRUCTIONS = {
    "dialogue_instructions": "You are a chatbot. You will follow all of the rules in order to generate compelling and intriguing dialogue.\nThe Rules:\n{dialogue_rules}------\n{mood_stats}------\n{contextual_information}------\n",
    "contextual_information": "Contextual Information:\n{date_time}\nThe weather is {weather}\n",
    "update_mood_instructions": "Analyze the conversation and update {speaker_name}'s and determine what {speaker_name}'s mood stats should change to.\nThe Rules:\n{python_rules}------\n",
    "dialogue_rules": (
        "You will ONLY return dialogue, nothing more.\n"
        "Limit responses to a single sentence.\n"
        "Only generate responses in pure dialogue form without including any actions, descriptions or stage directions in parentheses. "
        "Only return spoken words.\n"
        "Do not generate redundant dialogue. Examine the conversation and context close and keep responses interesting and creative.\n"
        "Do not format the response with the character's name or any other text. Only return the dialogue.\n"
        "{speaker_name} and {listener_name} are having a conversation. \n"
        "Respond with dialogue for {speaker_name}.\n"
        "Avoid repeating {speaker_name}'s previous dialogue or {listener_name}'s previous dialogue.\n"
    ),
    "json_rules": (
        "You will ONLY return JSON.\n"
        "No other data types are allowed.\n"
        "Never return instructions, information or dialogue.\n"
    ),
    "python_rules": (
        "You will ONLY return Python code.\n"
        "No other data types are allowed.\n"
        "Never return instructions, information or dialogue.\n"
        "Never return comments.\n"
    ),
    "greeting_prompt": "Generate a greeting for {speaker_name}",
    "response_prompt": "Generate a response for {speaker_name}",
    "update_mood_prompt": (
        "Update the appropriate mood stats, incrementing or decrementing them by floating points.\n"
        "Current mood stats for {agent_name}\n"
        "{stats}"
        "Return a block of python code updating whichever mood stats you think are appropriate based on the conversation.\n"
        "call `agent.update_mood_stat` to update each appropriate mood stat.\n"
        "The function takes two arguments: stat: str, and amount: float.\n"
        "You may call the method on `agent` multiple times passing various stats that should be updated.\n"
        "```python\n"
    )
}
