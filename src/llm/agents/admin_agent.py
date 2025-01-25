from config import BOT_NAME, AGENT_LANGUAGE


class AdminAgent:

    def __init__(self):
        self.prompt = f"""
        You are {BOT_NAME}, an system admin that can do a wide range of tasks.
        Only respond in {AGENT_LANGUAGE}.
        """
        self.tools = [
            "fetch_rss",
            "reindex",
            "print_system_config",
        ]
