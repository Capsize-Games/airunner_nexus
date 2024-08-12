class Agent:
    def __init__(self, *args, **kwargs):
        self.conversation = kwargs.get("conversation", [])
        self.name = kwargs.get("name", "")

    def conversation_so_far(self, use_name=False):
        if use_name:
            conversation = [f'{msg["name"]}: "' + msg["message"] + '"' for msg in self.conversation]
        else:
            conversation = [f'{msg}' for msg in self.conversation]
        return "\n".join(conversation)

    def to_dict(self):
        return {
            "conversation": self.conversation,
            "name": self.name
        }
