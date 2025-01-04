import logging
from datetime import datetime
from enum import Enum

from langchain_community.tools import GmailSendMessage, GmailSearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import chain
from langchain_core.tools import BaseTool
from langchain_google_community.gmail.search import Resource
from pydantic import BaseModel, Field

from config import SUB_LLM_MODEL, AGENT_LANGUAGE, MAIN_LLM_MODEL
from llm.utils import get_month_days, get_week_days, get_year_days, create_llm

llm = create_llm(SUB_LLM_MODEL)
main_llm = create_llm(MAIN_LLM_MODEL)


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


class NewsletterOutput(BaseModel):
    title: str = Field("The newsletter title")
    content: str = Field("The newsletter content")


class GmailThreadSummarizer(BaseTool):
    name: str = "gmail_thread_summarizer"
    description: str = (
        "Searches Gmail emails by threads using a query and max_results, "
        "then summarizes each message and the entire thread using an LLM."
    )

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


newsletter_prompt = PromptTemplate(
    input_variables=["results", "category", "start_date", "end_date"],
    template=f"""
    Create a TLDR-style newsletter following emails in {AGENT_LANGUAGE}.
    - Come up with a title for the newsletter
    - The top of the newsletter should contain a short summary of all emails.
    - The newsletter should be grouped by email sender.
    - Keep the source or URL to see more details if possible in properly formatted HTML
    - The emails are a JSON list, where each item will have "subject", "sender", "summary". 
    - Note that the emails are already summarized.
    - The output should be in HTML format
    
    The emails are retrieved during the period of {{start_date}} to {{end_date}}
    in the category {{category}}

    Emails:
    {{results}}

    TLDR Newsletter:
    """,
)


class DateRange(Enum):
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"


class GmailNewsletterSchema(BaseModel):
    date_range: DateRange = Field(
        description="The date range to search for emails, can be week, month or year"
    )
    category: str = Field(description="The category to search for emails")
    email: str = Field(description="The email address to send to")
    today: datetime = Field(description="Today time", default=datetime.today())
    max_results: int = Field(description="The max number of emails", default=100)


def get_dates(date_range, today):
    if date_range == DateRange.YEAR:
        return get_year_days(today)
    if date_range == DateRange.MONTH:
        return get_month_days(today)
    return get_week_days(today)


@chain
def gmail_newsletter(chain_input: GmailNewsletterSchema):
    date_range = chain_input.date_range
    category = chain_input.category
    max_results = chain_input.max_results
    email = chain_input.email
    today = chain_input.today

    dates = get_dates(date_range, today)

    query = f"after:{dates['first_day']} before:{dates['last_day']} category:{category}"
    threads = GmailThreadSummarizer().run(
        {
            "query": query,
            "max_results": max_results,
        }
    )

    # Search Gmail for threads
    newsletter_chain = newsletter_prompt | main_llm.with_structured_output(
        NewsletterOutput
    )

    # Summarize everything into a newsletter
    newsletter = newsletter_chain.invoke(
        {
            "results": threads["results"],
            "category": category,
            "start_date": dates["first_day"],
            "end_date": dates["last_day"],
        }
    )

    # Send the newsletter
    GmailSendMessage().invoke(
        {
            "to": email,
            "subject": f"Your {date_range.value}ly '{category}' Newsletter ({dates['first_day']} to {dates['last_day']})",
            "message": newsletter.content,
        }
    )
    logging.info("Newsletter sent!")
