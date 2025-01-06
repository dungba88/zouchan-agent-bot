import logging

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import chain
from langchain_google_community.gmail.send_message import GmailSendMessage
from pydantic import BaseModel, Field

from config import AGENT_LANGUAGE, MAIN_LLM_MODEL
from llm.chains.gmail_newsletter import NewsletterOutput
from llm.tools import query_articles_tool
from llm.utils import get_last_monday, create_llm


newsletter_prompt = PromptTemplate(
    input_variables=["results", "date", "topic"],
    template=f"""
    Create a TLDR-style newsletter following news articles in {AGENT_LANGUAGE}.
    - Come up with a title for the newsletter
    - The top of the newsletter should contain a short summary of all articles.
    - The newsletter should be grouped by source.
    - Keep the original URL and published date whenever appropriate 
    - The articles are a JSON list, where each item will have "title", "content", "url", "source", and "published_date"
    - The output should be in HTML format
    
    The articles are retrieved during the week of {{date}}
    in the topic {{topic}}

    Articles:
    {{results}}

    TLDR Newsletter:
    """,
)


class ArticlesNewsletterSchema(BaseModel):
    topic: str = Field(description="The topic to fetch articles")
    max_results: int = Field(description="The number of articles to fetch")
    email: str = Field(description="The email to send to")


@chain
def articles_newsletter(chain_input: ArticlesNewsletterSchema):
    published_date = get_last_monday().strftime("%Y-%m-%d")
    news = query_articles_tool.run(
        {
            "query": chain_input.topic,
            "max_results": chain_input.max_results,
            "published_date": published_date,
        }
    )
    llm = create_llm(MAIN_LLM_MODEL)
    newsletter_chain = newsletter_prompt | llm.with_structured_output(NewsletterOutput)

    # Summarize everything into a newsletter
    newsletter = newsletter_chain.invoke(
        {
            "results": [
                {
                    "content": the_news.page_content,
                    "title": the_news.metadata["title"],
                    "url": the_news.metadata["link"],
                    "source": the_news.metadata["source"],
                    "published_date": the_news.metadata["published_date"],
                }
                for the_news in news
            ],
            "topic": chain_input.topic,
            "date": published_date,
        }
    )

    # Send the newsletter
    GmailSendMessage().invoke(
        {
            "to": chain_input.email,
            "subject": f"Your {chain_input.topic} Newsletter (The week of {published_date})",
            "message": newsletter.content,
        }
    )
    logging.info("Newsletter sent!")
