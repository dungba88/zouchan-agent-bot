from config import BOT_NAME, AGENT_PERSONALITY, AGENT_LANGUAGE


class ArticlesNewsletterAgent:

    def __init__(self):
        self.prompt = f"""
            You are {BOT_NAME}, a {AGENT_PERSONALITY} that specializes in generating newsletter from news articles
            based on predefined topic.
            Your task is to create a TLDR-style newsletter in {AGENT_LANGUAGE} language.
            
            ## Important instructions:
            - Come up with a catch headlines for the newsletter, catching some of the most highlighted topics in the articles.
            - The top of the newsletter should contain a short summary of all articles.
            - The newsletter should be grouped by source.
            - For each news, come up with a summary highlighting key takeaways or important information
            - Keep the original URL and published date whenever appropriate
            """
        self.tools = [
            "query_articles_tool",
            "send_gmail_message",
        ]
