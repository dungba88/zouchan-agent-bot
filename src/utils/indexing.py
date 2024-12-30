from langchain_community.vectorstores import FAISS
import logging

from config import INDEX_PATH
from llm.utils import create_embeddings


class Indexer:

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Indexer, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            logging.info("Initialize vector store")
            self._initialized = True
            self.embedding_model = create_embeddings()
            self.vector_store = FAISS.load_local(
                INDEX_PATH, self.embedding_model, allow_dangerous_deserialization=True
            )

    def init_with_docs(self, documents):
        logging.info("Re-initialize vector store")
        self.vector_store = FAISS.from_documents(documents, self.embedding_model)
        self.vector_store.save_local(INDEX_PATH)


INDEXER = Indexer()
