from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool
from llama_index.core import Settings
import torch
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer
from typing import Iterator

class ChatLlamaRag:
    def __init__(self) -> None:
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
    
    def query(self, user_input: str) -> Iterator[str]:
        # Adiciona a mensagem do usuário à memória
        user_message = ChatMessage(role=MessageRole.USER, content=user_input)
        self.memory.put(user_message)
        
        # Obtém o histórico formatado corretamente
        chat_history = self.memory.get_all()
        
        # Gera resposta usando o método adequado para conversas
        response_gen = Settings.llm.stream_chat(chat_history)

        full_response = ""
        for chunk in response_gen:
            print(chunk.delta, end="", flush=True)
            full_response += chunk.delta
            yield full_response
        
        # Adiciona a resposta à memória
        assistant_message = ChatMessage(role=MessageRole.ASSISTANT, content=full_response)
        self.memory.put(assistant_message)
    
if __name__ == '__main__':

    aux = ChatLlamaRag()

    while True:
        user_input = input("Usuário: ")
        for resposta_parcial in aux.query(user_input):
            pass