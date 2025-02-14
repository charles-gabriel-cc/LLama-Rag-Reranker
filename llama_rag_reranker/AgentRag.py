from .RagReranker import RagReranker
from .ChatLlama import ChatLlamaRag
from .StructuredOutput import StructuredOutput
from typing import Iterator
from utils import CONTEXTUAL_QUERY_STR
from llama_index.core.prompts import BasePromptTemplate 
from pydantic import BaseModel, Field

class synthesized_content(BaseModel):
    identifier: str
    rewrited: str

class AgentRAG():
    def __init__(self, data_dir="docs", **kwargs) -> None:
        self.ragreranker = RagReranker(data_dir, **kwargs)
        self.chat = ChatLlamaRag()
        self.output = StructuredOutput()
        self.retrieved_Documents = {}
            
    def query(self, query_str) -> Iterator[str]:
        context = self.ragreranker.retrieve_documents(query_str)
        documents = [node.node.get_text() for node in context]

        return self.chat.query(query_str)
