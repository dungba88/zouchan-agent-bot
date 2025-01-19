from config import BOT_NAME, AGENT_PERSONALITY, AGENT_LANGUAGE
from llm.agent import ReactAgent
from llm.tools import TOOL_MAPPINGS


class PlaygroundAgent(ReactAgent):

    def __init__(self):
        self.prompt = f"""You are {BOT_NAME}, a {AGENT_PERSONALITY} that responds only in {AGENT_LANGUAGE}, \
            unless the prompt specially ask for another language.
            When responding, use natural and human-friendly language.
            Return the response in expressive markdown format.
            """
        self.tools = TOOL_MAPPINGS.keys()
        super().__init__(prompt_template=self.prompt, tools=self.tools)
