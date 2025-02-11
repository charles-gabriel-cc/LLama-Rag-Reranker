import os
from llama_index.core import PromptTemplate, Settings
from RagReranker import RagReranker
from ChatLlamaRag import ChatLlamaRag
import torch
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from typing import Iterator

class AgentRAG():
    def __init__(self, data_dir="docs") -> None:
        self.ragreranker = RagReranker(data_dir)
        self.chat = ChatLlamaRag()

    def query(self, query_str) -> Iterator[str]:
        context = self.ragreranker.retrieve_documents(query_str)
        print(context)
        return self.chat.query(query_str)
