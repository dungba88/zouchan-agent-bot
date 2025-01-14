from llm.agent import ReactAgent
from llm.agents.admin_agent import AdminAgent
from llm.agents.articles_newsletter_agent import ArticlesNewsletterAgent
from llm.agents.gmail_newsletter_agent import GmailNewsletterAgent
from llm.agents.playground_agent import PlaygroundAgent
from llm.agents.research_assistant_agent import ResearchAssistantAgent
from llm.agents.weather_assistant_agent import WeatherAssistantAgent


def create_agents():
    agent_configs = {
        "playground_agent": PlaygroundAgent(),
        "research_assistant_agent": ResearchAssistantAgent(),
        "articles_newsletter_agent": ArticlesNewsletterAgent(),
        "gmail_newsletter_agent": GmailNewsletterAgent(),
        "weather_assistant_agent": WeatherAssistantAgent(),
        "admin_agent": AdminAgent(),
    }
    return {
        name: ReactAgent(config.prompt, config.tools)
        for name, config in agent_configs.items()
    }
