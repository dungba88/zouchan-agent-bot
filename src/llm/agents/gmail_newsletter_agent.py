from typing import Type

from langchain_community.tools import GmailSearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool
from langchain_google_community.gmail.search import Resource
from pydantic import BaseModel, Field

from config import SUB_LLM_MODEL, AGENT_LANGUAGE, BOT_NAME, AGENT_PERSONALITY
from llm.utils import create_llm


llm = create_llm(SUB_LLM_MODEL)

EMAIL_SUMMARIZER_PROMPT = PromptTemplate(
    input_variables=["content"],
    template="""
            Summarize the following email, focusing on:
            - Important excerpts, events or information
            - Action items
            - Key takeaways
            - Keeping URL to see more details
            
            Email Content:
            {content}
            
            Summary:
            """,
)


class GmailThreadSummarizerSchema(BaseModel):
    query: str = Field(
        description="The Gmail query. Example filters include from:sender,"
        " to:recipient, subject:subject, -filtered_term,"
        " in:folder, is:important|read|starred, after:year/mo/date, "
        "before:year/mo/date, label:label_name"
        ' "exact phrase".'
        " Search newer/older than using d (day), m (month), and y (year): "
        "newer_than:2d, older_than:1y."
        " Attachments with extension example: filename:pdf. Multiple term"
        " matching example: from:amy OR from:david.",
    )
    max_results: int = Field(
        description="The max number of emails to return", default=100
    )


class GmailThreadSummarizer(BaseTool):
    name: str = "gmail_thread_summarizer"
    description: str = (
        "Searches Gmail emails by threads using a query and max_results, "
        "then summarizes each message and the entire thread using an LLM."
    )
    args_schema: Type[BaseModel] = GmailThreadSummarizerSchema

    @staticmethod
    def summarize_message(message: dict) -> str:
        """Summarize an individual email message."""
        message_content = message.get("body", "")
        if not message_content:
            return "No content to summarize."

        the_chain = EMAIL_SUMMARIZER_PROMPT | llm | StrOutputParser()
        return the_chain.invoke({"content": message_content})

    def _run(self, query: str, max_results: int = 100) -> dict:
        """
        Executes the tool:
        1. Searches for email threads matching the query.
        2. Summarizes each message and the entire thread.
        """
        threads = GmailSearch().run(
            tool_input={
                "query": query,
                "resources": Resource.THREADS,
                "max_results": max_results,
            }
        )

        results = []
        for thread in threads:
            thread_id = thread.get("id")
            thread_summary = self.summarize_message(thread)

            results.append(
                {
                    "id": thread_id,
                    "subject": thread["subject"],
                    "sender": thread["sender"],
                    "summary": thread_summary,
                }
            )

        return {
            "results": results,
        }


class GmailNewsletterAgent:

    def __init__(self):
        self.prompt = f"""
        You are {BOT_NAME}, a {AGENT_PERSONALITY} that specializes in generating TLDR newsletter from emails. 
        Your task is to create TLDR-style newsletter for emails in {AGENT_LANGUAGE}.
        - Come up with a title for the newsletter
        - Pay attention when searching for emails, you must absolutely follow all criteria given by users
          such as time range, category or sender.
        - The top of the newsletter should contain a short summary of all emails.
        - The newsletter should be grouped by email sender.
        - Keep the source or URL to see more details if possible
        """
        self.tools = [
            "gmail_thread_summarizer",
            "send_gmail_message",
        ]
