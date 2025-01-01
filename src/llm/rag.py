from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from config import SUB_LLM_MODEL
from llm.utils import create_llm
from utils.indexing import get_indexer_instance


class RagChain:

    def __init__(self):
        llm = create_llm(SUB_LLM_MODEL)
        # See full prompt at https://smith.langchain.com/hub/langchain-ai/retrieval-qa-chat
        retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

        combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
        self.rag_chain = create_retrieval_chain(
            get_indexer_instance().vector_store.as_retriever(), combine_docs_chain
        )

    def invoke(self, prompt):
        response = self.rag_chain.invoke({"input": prompt})
        return {
            "output": response["answer"],
            "context": [doc.dict() for doc in response["context"]],
        }
