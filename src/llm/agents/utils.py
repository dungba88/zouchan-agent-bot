from llm.agent import ReactAgent
from llm.agents.admin_agent import AdminAgent
from llm.agents.articles_newsletter_agent import ArticlesNewsletterAgent
from llm.agents.gmail_newsletter_agent import GmailNewsletterAgent
from llm.agents.playground_agent import PlaygroundAgent
from llm.agents.research_assistant_agent import ResearchAssistantAgent
from llm.agents.travel_assistant_agent import TravelAssistantAgent


def create_agents():
    agent_configs = {
        "playground_agent": PlaygroundAgent(),
        "research_assistant_agent": ResearchAssistantAgent(),
        "articles_newsletter_agent": ArticlesNewsletterAgent(),
        "gmail_newsletter_agent": GmailNewsletterAgent(),
        "travel_assistant_agent": TravelAssistantAgent(),
        "admin_agent": AdminAgent(),
    }
    return {
        name: ReactAgent(config.prompt, config.tools)
        for name, config in agent_configs.items()
    }
