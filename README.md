# LLama-Rag-Reranker

**LLama-Rag-Reranker** implements a **rag reranker** function for the **Llama Index** library. Additionally, the project provides a chatbot API that leverages this tool to enhance response quality in user interactions.

The project has been tested with **Python 3.10.0** and **CUDA 11.8**. To set up the project, follow these steps:

## Requirements

- **Python**: 3.10.0
- **CUDA**: 11.8
- - **OLLAMA**: 11.8

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/charles-gabriel-cc/LLama-Rag-Reranker.git
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. You need to install **PyTorch** with **CUDA 11.8** support
    ```bash
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

## Usage

The core functionality of **LLama-Rag-Reranker** is encapsulated in the `RagReranker` class. This class implements a retrieval-augmented generation (RAG) pipeline with reranking.

To use the framework it is **necessary** to define the default models to be used by the **LLama index** library, this is done by defining the `llm` and `embed_model` properties of the `Settings` variable, this can be done as follows:

```python
from llama_index.core import Settings

Settings.embed_model = HuggingFaceEmbedding(
    model_name="all-MiniLM-L6-v2"
)

Settings.llm = HuggingFaceLLM(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    tokenizer_name="meta-llama/Llama-3.2-3B-Instruct",
)
```

### Example

To use the `RagReranker` class, you must create a folder (e.g., `docs`) containing the documents you want to use for the RAG pipeline. You can then instantiate the class and call the `retrieve_documents` method with a query string. Below is an example of how to use the class:

1. Create a folder named `docs` (or any name you prefer) and place your documents inside it.
2. Use the following code to retrieve and rerank documents:

```python
from rag_reranker import RagReranker

# Initialize the RagReranker class
reranker = RagReranker(data_dir="path_to_docs")  # Replace with the path to your folder

# Retrieve and rerank documents based on a query
query = "What is retrieval-augmented generation?"
results = reranker.retrieve_documents(query)
```
