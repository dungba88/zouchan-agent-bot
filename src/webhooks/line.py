import base64
import hashlib
import hmac
import logging
import os
from datetime import datetime
from typing import Literal, Union, Optional, List
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse, quote

import requests
from flask import jsonify, request
from pydantic import BaseModel, Field

from config import SUB_LLM_MODEL
from llm.agent import ReactAgent, AgentInput
from llm.utils import create_llm


llm = create_llm(SUB_LLM_MODEL)
MAX_BUBBLE_LENGTH = 3


class ActionableContent(BaseModel):
    title: str = Field(..., description="The title of this actionable content")
    text: str = Field(..., description="The text of this actionable content")
    image_url: Optional[str] = Field(
        ..., description="The image URL of this actionable content"
    )
    url: Optional[str] = Field(..., description="The URL of this actionable content")


class LineMessage(BaseModel):
    text: str = Field(
        ...,
        description="The text content of the message, ignore any actionable contents",
    )
    actionable_contents: List[ActionableContent] = Field(
        ..., description="Actionable content, such as URL, image"
    )


class TextMessage(BaseModel):
    type: Literal["text"] = Field("text", description="Text message, fixed to 'text'.")
    text: str = Field(..., description="The text content of the message")


class FlexMessage(BaseModel):
    type: Literal["flex"] = Field(
        "flex", description="Type of the message, fixed to 'flex'."
    )
    altText: str = Field(
        ..., description="Alternative text for accessibility and unsupported devices."
    )
    contents: Union["Bubble", "Carousel"] = Field(
        ..., description="Contents of the flex message, either a bubble or a carousel."
    )


class Bubble(BaseModel):
    type: Literal["bubble"] = Field(
        "bubble", description="Type of the container, fixed to 'bubble'."
    )
    direction: Optional[Literal["ltr", "rtl"]] = Field(
        "ltr",
        description="Text direction, either left-to-right (ltr) or right-to-left (rtl).",
    )
    header: Optional["Box"] = Field(None, description="Header block of the bubble.")
    hero: Optional["Image"] = Field(None, description="Hero image block of the bubble.")
    body: Optional["Box"] = Field(None, description="Body block of the bubble.")
    footer: Optional["Box"] = Field(None, description="Footer block of the bubble.")


class Carousel(BaseModel):
    type: Literal["carousel"] = Field(
        "carousel", description="Type of the container, fixed to 'carousel'."
    )
    contents: List[Bubble] = Field(
        ..., description="List of bubble containers in the carousel."
    )


class Box(BaseModel):
    type: Literal["box"] = Field(
        "box", description="Type of the component, fixed to 'box'."
    )
    layout: Literal["horizontal", "vertical", "baseline"] = Field(
        ..., description="Box layout, determines how child components are arranged."
    )
    contents: List[
        Union["Text", "Image", "Button", "Spacer", "Filler", "Separator", "Box"]
    ] = Field(..., description="List of child components within the box.")
    spacing: Optional[Literal["none", "xs", "sm", "md", "lg", "xl", "xxl"]] = Field(
        None, description="Spacing between components within the box."
    )
    margin: Optional[Literal["none", "xs", "sm", "md", "lg", "xl", "xxl"]] = Field(
        None, description="Margin outside the box."
    )
    backgroundColor: Optional[str] = Field(
        None, description="Background color of the box."
    )
    borderColor: Optional[str] = Field(None, description="Border color of the box.")
    borderWidth: Optional[str] = Field(None, description="Border width of the box.")
    cornerRadius: Optional[str] = Field(None, description="Corner radius of the box.")
    width: Optional[str] = Field(None, description="Width of the box.")
    height: Optional[str] = Field(None, description="Height of the box.")
    paddingAll: Optional[str] = Field(
        None, description="Padding applied to all sides of the box."
    )
    paddingTop: Optional[str] = Field(
        None, description="Padding applied to the top side of the box."
    )
    paddingBottom: Optional[str] = Field(
        None, description="Padding applied to the bottom side of the box."
    )
    paddingStart: Optional[str] = Field(
        None, description="Padding applied to the start side of the box."
    )
    paddingEnd: Optional[str] = Field(
        None, description="Padding applied to the end side of the box."
    )


class Text(BaseModel):
    type: Literal["text"] = Field(
        "text", description="Type of the component, fixed to 'text'."
    )
    text: str = Field(..., description="Text content.")
    size: Optional[
        Literal["xxs", "xs", "sm", "md", "lg", "xl", "xxl", "3xl", "4xl", "5xl"]
    ] = Field(None, description="Font size of the text.")
    align: Optional[Literal["start", "end", "center"]] = Field(
        None, description="Text alignment."
    )
    gravity: Optional[Literal["top", "center", "bottom"]] = Field(
        None, description="Gravity of the text within the box."
    )
    wrap: Optional[bool] = Field(
        True, description="Whether the text should wrap if it overflows."
    )
    weight: Optional[Literal["regular", "bold"]] = Field(
        None, description="Font weight of the text."
    )
    color: Optional[str] = Field(None, description="Text color.")
    decoration: Optional[Literal["none", "underline", "line-through"]] = Field(
        None, description="Text decoration."
    )


class Image(BaseModel):
    type: Literal["image"] = Field(
        "image", description="Type of the component, fixed to 'image'."
    )
    url: str = Field(..., description="URL of the image.")
    size: Optional[
        Literal["xxs", "xs", "sm", "md", "lg", "xl", "xxl", "3xl", "4xl", "5xl", "full"]
    ] = Field(None, description="Size of the image.")
    aspectRatio: Optional[
        Literal[
            "1:1",
            "1.51:1",
            "1.91:1",
            "4:3",
            "16:9",
            "20:13",
            "2:1",
            "3:1",
            "3:4",
            "9:16",
            "1:2",
            "1:3",
        ]
    ] = Field(None, description="Aspect ratio of the image.")
    aspectMode: Optional[Literal["cover", "fit"]] = Field(
        None, description="Aspect mode of the image."
    )
    backgroundColor: Optional[str] = Field(
        None, description="Background color of the image."
    )


class Separator(BaseModel):
    type: Literal["separator"] = Field(
        "separator", description="Type of the component, fixed to 'separator'."
    )
    margin: Optional[Literal["none", "xs", "sm", "md", "lg", "xl", "xxl"]] = Field(
        None, description="Margin around the separator."
    )
    color: Optional[str] = Field(None, description="Color of the separator.")


class Spacer(BaseModel):
    type: Literal["spacer"] = Field(
        "spacer", description="Type of the component, fixed to 'spacer'."
    )
    size: Optional[Literal["xs", "sm", "md", "lg", "xl", "xxl"]] = Field(
        None, description="Size of the spacer."
    )


class Filler(BaseModel):
    type: Literal["filler"] = Field(
        "filler", description="Type of the component, fixed to 'filler'."
    )


class Button(BaseModel):
    type: Literal["button"] = Field(
        "button", description="Type of the component, fixed to 'button'."
    )
    action: "Action" = Field(..., description="Action triggered by the button.")
    style: Optional[Literal["link", "primary", "secondary"]] = Field(
        None, description="Style of the button."
    )
    color: Optional[str] = Field(None, description="Color of the button.")
    height: Optional[Literal["sm", "md"]] = Field(
        None, description="Height of the button."
    )
    gravity: Optional[Literal["top", "center", "bottom"]] = Field(
        None, description="Gravity of the button within the box."
    )


class Action(BaseModel):
    type: Literal["uri"] = Field(..., description="Type of action.")
    label: Optional[str] = Field(None, description="Label for the action.")
    data: Optional[str] = Field(None, description="Data carried with the action.")
    uri: Optional[str] = Field(None, description="URI for the action (for 'uri' type).")


# Update forward references
FlexMessage.update_forward_refs()
Bubble.update_forward_refs()
Carousel.update_forward_refs()
Box.update_forward_refs()
Text.update_forward_refs()
Image.update_forward_refs()
Button.update_forward_refs()


def url_encode_params(uri):
    # Parse the URI into components
    parsed_url = urlparse(uri)

    encoded_path = quote(parsed_url.path)

    # Parse the query parameters
    query_params = parse_qs(parsed_url.query)

    # URL encode each parameter value
    encoded_query = urlencode({k: v[0] for k, v in query_params.items()})

    # Rebuild the URI with the encoded parameters
    updated_url = parsed_url._replace(path=encoded_path, query=encoded_query)
    return urlunparse(updated_url)


def handle_webhook_event(agent):
    channel_secret = os.environ.get("LINE_CHANNEL_SECRET")
    # Validate the X-Line-Signature header
    signature = request.headers.get("X-Line-Signature")
    body = request.get_data(as_text=True)
    if not verify_signature(channel_secret, body, signature):
        return jsonify({"error": "Invalid signature"}), 400

    # Parse the webhook payload
    payload = request.json
    if not payload:
        return jsonify({"error": "Invalid payload"}), 400

    user_messages = {}
    # Process events
    events = payload.get("events", [])
    for event in events:
        source = event.get("source", {})
        message = event.get("message", {})

        # only accepts user messages
        if source.get("type") != "user":
            continue
        # only accept text messages
        if event.get("type") != "message" or message.get("type") != "text":
            continue

        user_id = source.get("userId")

        if user_id not in user_messages:
            user_messages[user_id] = []
        user_messages[user_id].append(message)

    for user_id, messages in user_messages.items():
        handle_text_messages(agent, user_id, messages)

    return jsonify({"message": "Webhook received successfully"}), 200


def build_actionable_item_text(item):
    contents = [
        Text(text=item.title, weight="bold", size="lg"),
    ]
    contents += [Text(text=the_text) for the_text in item.text.split("\n") if the_text]
    if item.url:
        contents.append(
            Button(
                action=Action(
                    type="uri",
                    label="View details",
                    uri=url_encode_params(item.url),
                )
            )
        )
    return Box(layout="vertical", contents=contents)


def build_actionable_item_box(item: ActionableContent):
    contents = [Separator(margin="md")]
    if item.image_url is not None:
        contents.append(Image(url=item.image_url, size="full"))
    contents.append(build_actionable_item_text(item))
    return Box(
        layout="vertical",
        contents=contents,
    )


def build_bubble_message(llm_output: LineMessage):
    return Bubble(
        body=Box(layout="horizontal", contents=[Text(text=llm_output.text)]),
        footer=Box(
            layout="vertical",
            contents=[
                build_actionable_item_box(item)
                for item in llm_output.actionable_contents
            ],
        ),
    )


def build_carousel_message(actionable_contents: List[ActionableContent]):
    return Carousel(
        contents=[
            Bubble(
                hero=Image(url=item.image_url, size="full") if item.image_url else None,
                header=Box(
                    layout="horizontal",
                    contents=[Text(text=item.title, weight="bold", size="lg")],
                ),
                body=Box(
                    layout="vertical",
                    contents=[
                        Text(text=the_text)
                        for the_text in item.text.split("\n")
                        if the_text
                    ],
                ),
                footer=(
                    Box(
                        layout="horizontal",
                        contents=[
                            Button(
                                action=Action(
                                    type="uri",
                                    label="View details",
                                    uri=url_encode_params(item.url),
                                )
                            )
                        ],
                    )
                    if item.url
                    else None
                ),
            )
            for item in actionable_contents
        ]
    )


def format_output(agent_response):
    output = agent_response["output"]
    # if there is no link and no image
    if output.find("http") == -1 and output.find("![") == -1:
        return [TextMessage(text=output.replace("**", "*"))]
    llm_output: LineMessage = llm.with_structured_output(LineMessage).invoke(
        f"""
        Convert the following response to LINE message format
        
        Response:
        {output}
    """
    )
    if not llm_output.actionable_contents:
        return [TextMessage(text=llm_output.text)]
    if len(llm_output.actionable_contents) <= MAX_BUBBLE_LENGTH:
        return [
            FlexMessage(
                altText=llm_output.text[:400], contents=build_bubble_message(llm_output)
            )
        ]
    return [
        TextMessage(text=llm_output.text),
        FlexMessage(
            altText=llm_output.text[:400],
            contents=build_carousel_message(llm_output.actionable_contents),
        ),
    ]


def handle_text_messages(agent: ReactAgent, user_id, messages):
    user_message = "\n".join([message.get("text") for message in messages])
    # TODO: Use conversation summary instead
    thread_id = f"{user_id}/line/{datetime.today().strftime("%Y-%m-%d")}"

    # Get the response from the LangChain agent
    try:
        agent_response = agent.invoke(
            AgentInput(prompt=user_message, thread_id=thread_id)
        )
        formatted_output = format_output(agent_response)

        # Send the response back to the user
        send_line_message(user_id, formatted_output)
    except Exception as e:
        logging.error(f"Error in LangChain agent: {e}")
        send_line_message(
            user_id,
            [TextMessage(text="Sorry, I couldn't process your message right now.")],
        )


def verify_signature(channel_secret, body, signature):
    """
    Verify the signature using the LINE Channel Secret.
    """
    hash = hmac.new(
        channel_secret.encode("utf-8"), body.encode("utf-8"), hashlib.sha256
    ).digest()
    expected_signature = base64.b64encode(hash).decode("utf-8")
    return hmac.compare_digest(expected_signature, signature)


def send_line_message(user_id, messages):
    """
    Send a message to a LINE user.
    """
    channel_access_token = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN")
    url = "https://api.line.me/v2/bot/message/push"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {channel_access_token}",
    }
    messages = [message.dict(exclude_none=True) for message in messages]
    body = {"to": user_id, "messages": messages}
    response = requests.post(url, headers=headers, json=body)
    if response.status_code != 200:
        logging.error(f"Error sending message: {response.status_code}, {response.text}")
        logging.error(messages)
