from utils.logging_utils import setup_logging


# setup logging before anything else
setup_logging()

import os
import logging

from config import STATIC_RESOURCES_PATH, BOT_NAME
from flask import Flask, request, jsonify, render_template, abort, send_from_directory

from llm.agents.utils import create_agents
from utils.cronservices import CronService


app = Flask(__name__)

agents = create_agents()
logging.info(f"Agents created: {agents.keys()}")

cron = CronService()
cron.run()
logging.info("Started cron service")


@app.route("/invoke-agent/<path:agent>", methods=["GET"])
def call_react_agent(agent):
    prompt = request.args.get("prompt")
    thread_id = request.args.get("thread_id")
    response = agents[agent].invoke(prompt, thread_id)
    return jsonify({"response": response, "status": "success"}), 200


# Route to render the HTML page
@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html", bot_name=BOT_NAME, agents=agents.keys())


# Route to render the HTML page
@app.route("/register")
def register():
    return render_template("register.html", bot_name=BOT_NAME)


# Route to render the HTML page
@app.route("/test")
def test():
    return render_template("test.html", bot_name=BOT_NAME)


@app.route("/resources/<path:filename>", methods=["GET"])
def serve_static_resource(filename):
    """
    Serves static resources from disk, validating the filename to prevent path injection.
    """
    # Validate the file path to prevent path traversal attacks
    requested_path = os.path.join(STATIC_RESOURCES_PATH, filename)
    safe_path = os.path.abspath(requested_path)
    base_dir = os.path.abspath(STATIC_RESOURCES_PATH)

    # Ensure the requested file is within the allowed directory
    if not safe_path.startswith(base_dir):
        abort(403, "Access denied")

    # Check if the file exists
    if not os.path.isfile(safe_path):
        abort(404, "File not found")

    # Serve the file
    return send_from_directory(STATIC_RESOURCES_PATH, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
