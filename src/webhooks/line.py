import base64
import hashlib
import hmac
import logging
import os
from datetime import datetime

import requests
from flask import jsonify, request

from llm.agent import ReactAgent


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


def handle_text_messages(agent: ReactAgent, user_id, messages):
    user_message = "\n".join([message.get("text") for message in messages])
    # TODO: Use conversation summary instead
    thread_id = f"{user_id}/line/{datetime.today().strftime("%Y-%m-%d")}"

    # Get the response from the LangChain agent
    try:
        agent_response = agent.invoke(user_message, thread_id)

        # Send the response back to the user
        send_line_message(user_id, agent_response["output"])
    except Exception as e:
        logging.error(f"Error in LangChain agent: {e}")
        send_line_message(user_id, "Sorry, I couldn't process your message right now.")


def verify_signature(channel_secret, body, signature):
    """
    Verify the signature using the LINE Channel Secret.
    """
    hash = hmac.new(
        channel_secret.encode("utf-8"), body.encode("utf-8"), hashlib.sha256
    ).digest()
    expected_signature = base64.b64encode(hash).decode("utf-8")
    return hmac.compare_digest(expected_signature, signature)


def send_line_message(user_id, message):
    """
    Send a message to a LINE user.
    """
    channel_access_token = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN")
    url = "https://api.line.me/v2/bot/message/push"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {channel_access_token}",
    }
    body = {"to": user_id, "messages": [{"type": "text", "text": message}]}
    response = requests.post(url, headers=headers, json=body)
    if response.status_code != 200:
        logging.error(f"Error sending message: {response.status_code}, {response.text}")
