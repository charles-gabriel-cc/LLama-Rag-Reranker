from RagReranker import RagReranker
from ChatLlamaRag import ChatLlamaRag
from typing import Iterator

class AgentRAG():
    def __init__(self, data_dir="docs") -> None:
        self.ragreranker = RagReranker(data_dir)
        self.chat = ChatLlamaRag()

    def query(self, query_str) -> Iterator[str]:
        context = self.ragreranker.retrieve_documents(query_str)
        return self.chat.query(query_str)
