# Zou-chan Agent Bot

A helpful bot backed by LangGraph

## Install

First install `pytorch` following the instruction at https://pytorch.org/get-started/locally. This is required for HuggingFace embeddings.

Then install the rest of dependencies:

```shell
pip install -r requirements.txt
cd src/
python setup.py
```

Other installations:

```shell
# if you use llama
ollama pull llama3.2

# install playwright for URL scrapers
playwright install
```

## Configuration

Rename `.env.sample` to `.env` and configure your API keys for various services in `.env` file

See other configurations in `src/config.py`

## Run

```shell
python main.py
```
