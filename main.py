import torch
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from AgentRag import AgentRAG
from llama_index.core import PromptTemplate, Settings

Settings.embed_model = HuggingFaceEmbedding(
    model_name="all-MiniLM-L6-v2"
)

Settings.llm = HuggingFaceLLM(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    tokenizer_name="meta-llama/Llama-3.2-3B-Instruct",
    device_map="cuda:0",
    model_kwargs={
        "torch_dtype": torch.float16,
    },
    max_new_tokens=100
)

if __name__ == '__main__':

    aux = AgentRAG()

    while True:
        user_input = input("Usuário: ")
        for resposta_parcial in aux.query(user_input):
            pass
