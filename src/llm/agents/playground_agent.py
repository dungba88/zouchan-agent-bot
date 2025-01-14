from config import PROMPT_TEMPLATE
from llm.agent import ReactAgent
from llm.tools import TOOL_MAPPINGS


class PlaygroundAgent(ReactAgent):

    def __init__(self):
        self.prompt = PROMPT_TEMPLATE
        self.tools = TOOL_MAPPINGS.keys()
        super().__init__(prompt_template=self.prompt, tools=self.tools)
