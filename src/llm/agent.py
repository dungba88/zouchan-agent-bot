import logging
import uuid
from datetime import datetime
from typing import Optional, Any

from langchain_core.messages import message_to_dict
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.utils import Output
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from config import MAIN_LLM_MODEL, USE_SHORT_TERM_MEMORY
from llm.tools import TOOL_MAPPINGS
from llm.utils import create_llm


class AgentInput(BaseModel):
    prompt: str = Field(..., description="The input prompt")
    thread_id: str = Field(None, description="The thread_id for short-term memory")


class BaseAgent(Runnable):

    def __init__(self, prompt_template, tools, format="markdown"):
        self.llm = create_llm(MAIN_LLM_MODEL)
        # Initialize the memory for short-term memory (conversation history)
        self.memory = self.initialize_memory()
        # Initialize tools for the agent
        disabled_tools = [tool for tool in tools if tool not in TOOL_MAPPINGS]
        tools = [TOOL_MAPPINGS[tool] for tool in tools if tool in TOOL_MAPPINGS]
        if disabled_tools:
            logging.warning(f"Disabled tools: {disabled_tools}")
        self.react_agent = create_react_agent(
            self.llm,
            tools=tools,
            checkpointer=self.memory,
            state_modifier=f"{prompt_template}. Return in {format} format",
        )

    def invoke(
        self, input: AgentInput, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Output:
        pass

    @staticmethod
    def initialize_memory():
        if USE_SHORT_TERM_MEMORY:
            return MemorySaver()
        return None


class ReactAgent(BaseAgent):

    def __init__(self, prompt_template, tools):
        super().__init__(prompt_template, tools)

    def invoke(
        self, input: AgentInput, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Output:
        if input.thread_id is None:
            input.thread_id = f"system/{uuid.uuid4()}"

        logging.info(f"Using thread_id: {input.thread_id}")

        prompt = f"""
        Current time is {datetime.today().strftime("%Y-%m-%d %H:%M:%S")}.

        {input.prompt}
        """

        inputs = {"messages": [("user", prompt)]}
        if config is None:
            config = RunnableConfig()
        config["configurable"] = {
            "thread_id": input.thread_id,
        }
        stream = self.react_agent.stream(inputs, config, stream_mode="values")
        response = ReactAgent._print_and_get_response(stream)
        logging.info("Re-act Agent executed.")

        return response

    @staticmethod
    def _print_and_get_response(stream):
        messages = []
        for s in stream:
            message = s["messages"][-1]
            if isinstance(message, tuple):
                print(message)
            else:
                message.pretty_print()
            messages.append(message)
        return {
            "messages": [
                message_to_dict(message)
                for message in messages[:-1]
                if message.type != "human"
            ],
            "output": message_to_dict(messages[-1])["data"]["content"],
        }
