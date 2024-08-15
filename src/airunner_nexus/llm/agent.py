class Agent:
    def __init__(self, *args, **kwargs):
        self.conversation = kwargs.get("conversation", [])
        self.name = kwargs.get("name", "")
        self.mood_stats = {
            "happy": 0,
            "sad": 0,
            "neutral": 0,
            "angry": 0,
            "paranoid": 0,
            "anxious": 0,
            "excited": 0,
            "bored": 0,
            "confused": 0,
            "relaxed": 0,
            "curious": 0,
            "frustrated": 0,
            "hopeful": 0,
            "disappointed": 0,
        }
        self.user = None

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

    def update_mood_stat(self, stat: str, amount: float):
        if stat in self.mood_stats:
            self.mood_stats[stat] += amount
