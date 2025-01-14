from datetime import datetime

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import chain
from langchain_google_community.gmail.send_message import GmailSendMessage
from pydantic import BaseModel, Field

from config import AGENT_LANGUAGE, MAIN_LLM_MODEL
from llm.tools import get_weather
from llm.utils import create_llm

PROMPT = PromptTemplate(
    input_variables=["request", "weather"],
    template=f"""
        You are a helpful weather assistant that responds only in {AGENT_LANGUAGE}.
        You are to provide the users a helpful tips regarding to the weather
        and the details of the weather in HTML format.
        
        The request of the user:
        {{request}}
        
        The 7-days weather data is:
        {{weather}}
        
        The current time is {datetime.today().strftime("%Y-%m-%d %H:%M:%S")}
        """,
)


class WeatherNotifySchema(BaseModel):
    lat: float = Field("Latitude to check the weather")
    lon: float = Field("Longitude to check the weather")
    request: str = Field("Request details to add to the prompt")


@chain
def weather_notify(chain_input: WeatherNotifySchema):
    weather = get_weather.invoke({
        "lat": chain_input.lat,
        "lon": chain_input.lon,
    })

    weather_notify_chain = PROMPT | create_llm(MAIN_LLM_MODEL)

    notification = weather_notify_chain.invoke(
        {
            "weather": weather,
            "request": chain_input.request,
        }
    )
    GmailSendMessage().invoke(
        {
            "to": "dungba.sg@gmail.com",
            "subject": "Weather Notification",
            "message": notification.content,
        }
    )
