from config import AGENT_LANGUAGE, BOT_NAME


class WeatherAssistantAgent:

    def __init__(self):
        self.prompt = f"""
        You are {BOT_NAME}, a helpful weather assistant that responds only in {AGENT_LANGUAGE}.
        Your task is to provide the users a helpful tips regarding to the weather
        and the details of the weather.
        - Default output (unless specified by user): Markdown
        """
        self.tools = [
            "get_weather",
            "send_gmail_message",
        ]
