from utils.logging_utils import setup_logging

# setup logging before anything else
setup_logging()

import os

from config import STATIC_RESOURCES_PATH, BOT_NAME
from flask import Flask, request, jsonify, render_template, abort, send_from_directory
import json
import logging

from llm.agent import ReactAgent, PlanExecuteAgent
from llm.chains.utils import CHAINS, run_chain
from llm.rag import RagChain
from llm.tools import TOOLS, ADMIN_TOOLS
from utils.cronservices import CronService


app = Flask(__name__)

react_agent = ReactAgent(tools=TOOLS + ADMIN_TOOLS)
logging.info("Created re-act agent")
plex_agent = PlanExecuteAgent(tools=TOOLS + ADMIN_TOOLS)
logging.info("Created plan-execute agent")
rag = RagChain()
logging.info("Created RAG chain")

cron = CronService()
cron.run()
logging.info("Started cron service")


@app.route("/react-agent", methods=["GET"])
def call_react_agent():
    prompt = request.args.get("prompt")
    thread_id = request.args.get("thread_id")
    response = react_agent.invoke(prompt, thread_id)
    return jsonify({"response": response, "status": "success"}), 200


@app.route("/plex-agent", methods=["GET"])
def call_plex_agent():
    prompt = request.args.get("prompt")
    response = plex_agent.invoke(prompt)
    return jsonify({"response": response, "status": "success"}), 200


@app.route("/rag", methods=["GET"])
def call_rag():
    prompt = request.args.get("prompt")
    response = rag.invoke(prompt)
    return jsonify({"response": response, "status": "success"}), 200


# Route to render the HTML page
@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html", bot_name=BOT_NAME)


@app.route("/chain")
def chain():
    return render_template(
        "chain.html",
        chains=[name for name in CHAINS.keys()],
        chain_schema={
            name: value.input_schema.model_json_schema()
            for name, value in CHAINS.items()
        },
    )


@app.route("/invoke-chain")
def invoke_chain():
    chain_name = request.args.get("chain")
    chain_input = json.loads(request.args.get("input"))
    response = run_chain(chain_name, chain_input)
    return jsonify({"response": response, "status": "success"}), 200


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
    app.run(debug=True)
