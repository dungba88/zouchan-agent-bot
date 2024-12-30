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

## Configuration

Configure your API keys for various services, such as in `.bashrc`

```shell
# if you use Cohere
export COHERE_API_KEY="..."

# if you use DeepSeek
export DEEPSEEK_API_KEY="..."

# if you use OpenAI
export OPENAI_API_KEY="..."

# if you use LangSmith
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
export LANGCHAIN_API_KEY="..."
export LANGCHAIN_PROJECT="..."

# if you use Tavily
export TAVILY_API_KEY="..."
```

See other configurations in `src/config.py`

## Run

```shell
python main.py
```
