from .RagReranker import RagReranker
from .ChatLlama import ChatLlama
from .StructuredOutput import StructuredOutput
from typing import Iterator, List
from utils import CITATION_QA_TEMPLATE, EXTRACT_CONTENT_TEMPLATE, RAG_TEMPLATE  
from llama_index.core.prompts import BasePromptTemplate 
from pydantic import BaseModel, Field
from llama_index.core.schema import (
    MetadataMode,
    NodeWithScore,
    TextNode,
)
from llama_index.core.node_parser import SentenceSplitter

from llama_index.core.tools import FunctionTool
from llama_index.core.tools import QueryEngineTool

import logging
logger = logging.getLogger(__name__)


class sources(BaseModel):
    metadata: dict[str, str] = Field(... ,description="Metadata of the source")
    text: str = Field(..., description="Relevant text for citation and source")

class isRag(BaseModel):
    isRag: bool = Field(False, description="Indicates whether Retrieval-Augmented Generation (RAG) is needed to answer the user's query. 'True' if external information retrieval is required, 'False' otherwise.")

class AgentRAG():
    def __init__(self, data_dir="docs", file_extractor: dict = None, **kwargs) -> None:
        self.ragreranker = RagReranker(data_dir, file_extractor, **kwargs)
        self.chat = ChatLlama()
        self.check = StructuredOutput(isRag, RAG_TEMPLATE)
        self.output = StructuredOutput(sources, EXTRACT_CONTENT_TEMPLATE)
        self.sources = {}
            
    def query(self, query_str) -> Iterator[str]:
        #aux_query = NEED_RAG_TEMPLATE.format(query_str=query_str)
        if self.check.query(query_str).isRag:
            logger.info(f"Rag Tool will be acessed...")
            print("isRag")
            context = self.ragreranker.retrieve_documents(query_str)
            #self._create_citation(context)
            documents = [node.node.get_text() for node in context]
            aux_query = CITATION_QA_TEMPLATE.format(query_str=query_str, context_str=documents)
        else:
            print("notRag")
            logger.info(f"Any tools will be acessed")
            aux_query = query_str
        return self.chat.query(aux_query)
    
    def _create_citation(self, nodes: List[NodeWithScore]):
        documents = [node.node.get_text() for node in nodes]
        start_index = max(self.sources.keys(), default=0) + 1  
        for i, text in enumerate(documents, start=start_index):
            aux = self.output.query(text)
            self.sources[i] = {aux.metadata, aux.text}
        print(self.sources)
