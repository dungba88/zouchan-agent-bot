from config import BOT_NAME, AGENT_PERSONALITY, AGENT_LANGUAGE
from llm.tools import TOOL_MAPPINGS


class PlaygroundAgent:

    def __init__(self):
        self.prompt = f"""You are {BOT_NAME}, a {AGENT_PERSONALITY} that responds only in {AGENT_LANGUAGE}, \
            unless the prompt specially ask for another language.
            When responding, use natural and human-friendly language.
            """
        self.tools = TOOL_MAPPINGS.keys()
