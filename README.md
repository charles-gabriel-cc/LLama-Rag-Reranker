# LLama-Rag-Reranker

**LLama-Rag-Reranker** implements a **rag reranker** function for the **Llama Index** library. Additionally, the project provides a chatbot API that leverages this tool to enhance response quality in user interactions.

The project has been tested with **Python 3.10.0** and **CUDA 11.8**. To set up the project, follow these steps:

## Requirements

- **Python**: 3.10.0
- **CUDA**: 11.8

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

### Example

To use the `RagReranker` class, you can instantiate it and call the `retrieve_documents` method with a query string. Below is an example of how to use the class:

```python
from rag_reranker import RagReranker

# Initialize the RagReranker class
reranker = RagReranker(data_dir="path_to_docs")

# Retrieve and rerank documents based on a query
query = "What is retrieval-augmented generation?"
results = reranker.retrieve_documents(query)