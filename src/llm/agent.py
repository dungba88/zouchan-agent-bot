import logging
import operator
import uuid
from datetime import datetime
from typing import TypedDict, List, Annotated, Tuple, Union

from langchain_core.messages import message_to_dict
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from config import MAIN_LLM_MODEL, USE_SHORT_TERM_MEMORY
from llm.tools import TOOL_MAPPINGS
from llm.utils import create_llm


class BaseAgent:

    def __init__(self, prompt_template, tools):
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
            state_modifier=prompt_template,
        )

    @staticmethod
    def initialize_memory():
        if USE_SHORT_TERM_MEMORY:
            return MemorySaver()
        return None


class ReactAgent(BaseAgent):

    def __init__(self, prompt_template, tools):
        super().__init__(prompt_template, tools)

    def invoke(self, prompt, thread_id=None):
        if thread_id is None:
            thread_id = f"system/{uuid.uuid4()}"

        logging.info(f"Using thread_id: {thread_id}")

        prompt = f"""
        Current time is {datetime.today().strftime("%Y-%m-%d %H:%M:%S")}.

        {prompt}
        """

        inputs = {"messages": [("user", prompt)]}
        config = {
            "configurable": {
                "thread_id": thread_id,
            },
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


class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )


class PlanExecute(TypedDict):
    input: str
    role: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str


class Response(BaseModel):
    """Response to user."""

    response: str


class Act(BaseModel):
    """Action to perform."""

    response: Response = Field(
        description="Only set this for the final response to users.\
                    If you need to further use tools to get the answer, use Plan."
    )
    plan: Plan = Field(
        description="Only set this if you need to further use tools to get the answer"
    )


def to_structured_output(llm, schema):
    # output_parser = PydanticOutputParser(pydantic_object=schema)
    # return llm.bind_tools(TOOLS + ADMIN_TOOLS, format=schema.model_json_schema()) | output_parser
    return llm.with_structured_output(schema)


class PlanExecuteAgent(BaseAgent):

    def __init__(self, prompt_template, tools):
        super().__init__(prompt_template, tools)
        self.planner = self._create_planner()
        self.replanner = self._create_replanner()
        self.graph = self._create_graph()

    def invoke(self, prompt):
        inputs = {"input": prompt}
        stream = self.graph.stream(inputs, stream_mode="values")
        response = PlanExecuteAgent._print_and_get_response(stream)
        logging.info("Plan-and-execute Agent executed.")

        return response

    @staticmethod
    def _print_and_get_response(stream):
        events = []
        for event in stream:
            for k, v in event.items():
                if k != "__end__":
                    logging.info(v)
            events.append(event)

        return {
            "events": events,
            "output": events[-1]["response"],
        }

    def _create_planner(self):
        planner_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """For the given objective, come up with a simple step by step plan. \
                    This plan should involve individual tasks, that if executed correctly will yield \
                    the correct answer. Do not add any superfluous steps. \
                    The result of the final step should be the final answer. Make sure that each step \
                    has all the information needed - do not skip steps.""",
                ),
                ("placeholder", "{messages}"),
            ]
        )
        # TODO: Use a different LLM here
        return planner_prompt | to_structured_output(self.llm, Plan)

    def _create_replanner(self):
        replanner_prompt = ChatPromptTemplate.from_template(
            """For the given objective, come up with a simple step by step plan. \
            This plan should involve individual tasks, that if executed correctly will \
            yield the correct answer. Do not add any superfluous steps. \
            The result of the final step should be the final answer. Make sure that each \
            step has all the information needed - do not skip steps.
            
            Your objective was this:
            {input}
            
            Your original plan was this:
            {plan}
            
            You have currently done the follow steps:
            {past_steps}
            
            - If there are more steps needed to achieve the objective or you need to change the original plan, return a **plan** with the remaining steps.
            - If all necessary steps have been completed, return a **response** to the user based on the information gathered.
            - Do **not** include any steps that have already been completed in the new plan.
            - Do **not** return an empty plan; if no further steps are needed, you **must** return a **response**.
            - Ensure your output is in the correct structured format as per the `Act` model.

            **Remember**:
            - The `Act` can contain either `plan` or `response`.
            - A `plan` contains a list of steps that still need to be done.
            - A `response` contains the **final answer** to the user.
            """
        )

        # TODO: Use a different LLM here
        return replanner_prompt | to_structured_output(self.llm, Act)

    def _create_graph(self):
        def execute_step(state: PlanExecute):
            plan = state["plan"]
            plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
            task = plan[0]
            task_formatted = f"""For the following plan:
                {plan_str}\n\nYou are tasked with executing step {1}, {task}."""
            agent_response = self.react_agent.invoke(
                {"messages": [("user", task_formatted)]}
            )
            return {
                "role": "agent",
                "past_steps": [(task, agent_response["messages"][-1].content)],
            }

        def plan_step(state: PlanExecute):
            plan = self.planner.invoke({"messages": [("user", state["input"])]})
            return {"plan": plan.steps, "role": "plan"}

        def replan_step(state: PlanExecute):
            output = self.replanner.invoke(state)
            if output.response is not None:
                return {"response": output.response.response, "role": "replan"}
            else:
                return {"plan": output.plan.steps, "role": "replan"}

        def should_end(state: PlanExecute):
            if "response" in state and state["response"]:
                return END
            else:
                return "agent"

        workflow = StateGraph(PlanExecute)
        workflow.add_node("planner", plan_step)
        workflow.add_node("agent", execute_step)
        workflow.add_node("replan", replan_step)

        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "agent")
        workflow.add_edge("agent", "replan")

        workflow.add_conditional_edges(
            "replan",
            # Next, we pass in the function that will determine which node is called next.
            should_end,
            ["agent", END],
        )
        return workflow.compile()
