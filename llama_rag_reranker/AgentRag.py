from .RagReranker import RagReranker
from .ChatLlama import ChatLlama
from .StructuredOutput import StructuredOutput
from typing import Iterator
from utils import CONTEXTUAL_QUERY_STR, NEED_RAG_TEMPLATE
from llama_index.core.prompts import BasePromptTemplate 
from pydantic import BaseModel, Field

class synthesized_content(BaseModel):
    identifier: str
    rewrited: str

class isRag(BaseModel):
    isRag: bool = Field(False, description="Indicates whether Retrieval-Augmented Generation (RAG) is needed to answer the user's query. 'True' if external information retrieval is required, 'False' otherwise.")

class AgentRAG():
    def __init__(self, data_dir="docs",file_extractor: dict = None, **kwargs) -> None:
        self.ragreranker = RagReranker(data_dir, file_extractor, **kwargs)
        self.chat = ChatLlama()
        self.check = StructuredOutput(isRag)
        self.output = StructuredOutput(synthesized_content)
        self.retrieved_Documents = {}
            
    def query(self, query_str) -> Iterator[str]:
        aux_query = NEED_RAG_TEMPLATE.format(query_str=query_str)
        if self.check.query_output(aux_query).isRag:
            print("isRag")
            context = self.ragreranker.retrieve_documents(query_str)
            documents = [node.node.get_text() for node in context]
        else:
            print("notRAG")
        return self.chat.query(query_str)
