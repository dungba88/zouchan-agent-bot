from langchain_community.vectorstores import FAISS

from config import INDEX_PATH
from llm.utils import create_embeddings
from utils.db import create_db


# initialize db
create_db()

# initialize vector store
vector_store = FAISS.from_texts(["hello world"], create_embeddings())
vector_store.save_local(INDEX_PATH)
