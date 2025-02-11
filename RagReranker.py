from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, QueryBundle, Document
import logging
from llama_index.core.postprocessor import LLMRerank
from llama_index.readers.json import JSONReader
import torch
from llama_index.core.retrievers import VectorIndexRetriever
from utils import get_valid_kwargs


#logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RagReranker:
    """
    A class that implements a retrieval-augmented generation (RAG) pipeline with reranking.

    This class is responsible for:
    1. Loading documents from a specified directory.
    2. Creating a vector-based index of documents.
    3. Implementing a retriever to fetch relevant documents based on similarity.
    4. Using a reranker to refine the retrieved results.

    Args:
        data_dir (str): The directory where the documents are stored. Defaults to "docs".
        **kwargs: Additional keyword arguments passed to specific methods for customization.
            SimpleDirectoryReader kwargs:
            - filename_as_id (bool): Uses the filename as the document ID.
            - num_files_limit (int): Sets the maximum number of files to read.
            - recursive (bool): If True, reads files from subdirectories recursively.
            - file_extractor (dict): Defines a custom extractor for specific file types.

            VectorStoreIndex kwargs:
            - show_progress (bool): Displays a progress bar when creating the index.
            - embed_model (EmbeddingModel): Embedding model used for indexing.
            - storage_context (StorageContext): Storage context for the index.

            VectorIndexRetriever kwargs:
            - similarity_top_k (int): Number of most similar documents to retrieve.

            LLMRerank kwargs:
            - choice_batch_size (int): Number of documents processed per batch during reranking.
            - top_n (int): Final number of documents returned after reranking.
            - rerank_model (LLMModel): Language model used for reranking.

    Methods:
        _reader(num_files_limit, recursive, file_extractor, filename_as_id, **kwargs):
            - Loads documents from `data_dir` using `SimpleDirectoryReader`.
            - Supports limiting file count, recursive search, and file format extraction.
        
        _create_index(show_progress, **kwargs):
            - Builds a vector-based index (`VectorStoreIndex`) from the loaded documents.
        
        _create_retriever(similarity_top_k, **kwargs):
            - Initializes a `VectorIndexRetriever` to retrieve top-K similar documents.
        
        _create_reranker(choice_batch_size, top_n, **kwargs):
            - Sets up an `LLMRerank` model to refine the retrieved documents.
        
        retrieve_documents(query, **kwargs):
            - Retrieves and reranks documents based on a given query string.
            - Uses `retriever` for initial retrieval and `reranker` for post-processing.
    """

    def __init__(self, data_dir: str = "docs", **kwargs) -> None:
        self.data_dir = data_dir
        self.documents: list[Document]
        self.index: VectorStoreIndex
        self.retriever: VectorIndexRetriever
        self.reranker: LLMRerank

        self._reader(**kwargs)
        self._create_index(**kwargs)
        self._create_retriever(**kwargs)
        self._create_reranker(**kwargs)

    def _reader(self, num_files_limit: int = 1 ,
                      recursive: bool = True, 
                      file_extractor: dict = None, 
                      filename_as_id: bool = True,
                      **kwargs) -> list[Document]:
        reader_kwargs = get_valid_kwargs(SimpleDirectoryReader, kwargs)
        try:
            reader = SimpleDirectoryReader(
                self.data_dir,
                file_extractor=file_extractor,
                num_files_limit=num_files_limit,
                recursive=recursive,
                filename_as_id=filename_as_id,
                **reader_kwargs
            )
            self.documents = reader.load_data()
        except:
            logger.info(f"Invalid argument in reader")
            raise 
        
    def _create_index(self, show_progress: bool = True,**kwargs) -> None:
        index_kwargs = get_valid_kwargs(VectorStoreIndex, kwargs)
        try:
            self.index = VectorStoreIndex.from_documents(
                self.documents,
                show_progress=show_progress,
                **index_kwargs
                )
        except:
            logger.info(f"Invalid argument in index")
            raise 

    def _create_retriever(self, 
                          similarity_top_k: int = 10, 
                          **kwargs) -> None:
        retriever_kwargs = get_valid_kwargs(VectorIndexRetriever, kwargs)
        try:
            self.retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=similarity_top_k,
                **retriever_kwargs
            )
        except:
            logger.info(f"Invalid argument in retriever")
            raise

    def _create_reranker(self,
                         choice_batch_size: int = 5,
                         top_n: int = 3,
                         **kwargs) -> None:
        reranker_kwargs = get_valid_kwargs(LLMRerank, kwargs)
        try:
            self.reranker = LLMRerank(
                choice_batch_size=choice_batch_size,
                top_n=top_n,    
                **reranker_kwargs 
            )
        except:
            logger.info(f"Invalid argument in reranker")
            raise
    
    def retrieve_documents(self, query: str, **kwargs) -> list[str]:
        try:
            query_bundle = QueryBundle(query)
            retrieved_nodes = self.retriever.retrieve(query_bundle)
            retrieved_nodes = self.reranker.postprocess_nodes(
                retrieved_nodes, query_bundle
            )
            documents = [node.node.get_text() for node in retrieved_nodes]
            logger.info(f"Successfully retrieved: {query}")
            return documents
        except Exception as e:
            logger.error(f"Error during retrieving: {str(e)}")
            raise

class EnhancedRagReranker(RagReranker):
    def _is_relevant(self, query: str, document_text: str) -> bool:
        """Classifica a relevância usando o LLM com prompt otimizado"""
        prompt = (
            f"[INSTRUÇÃO] Avalie se o texto abaixo contém informações relevantes para responder: '{query}'\n"
            f"[TEXTO] {document_text[:2000]}\n\n"
            "Responda APENAS com 'SIM' ou 'NÃO'.\n"
            "Considere relevante se abordar diretamente o tópico ou fornecer contexto útil."
        )
        
        try:
            response = Settings.llm.complete(prompt)
            return response.text.strip().lower() in ["sim", "yes", "y"]
        except Exception as e:
            logger.error(f"Falha na classificação: {str(e)}")
            return False  
        
    def retrieve_documents(self, query: str, **kwargs) -> list[str]:
        """Sobrescreve com etapa adicional de filtragem"""
        try:
            # Usa a implementação original até o rerank
            base_results = super().retrieve_documents(query, **kwargs)
            
            # Filtra documentos irrelevantes
            relevant_docs = [
                doc for doc in base_results
                if self._is_relevant(query, doc)
            ]
            
            logger.info(f"Relevância: {len(relevant_docs)}/{len(base_results)} documentos")
            return relevant_docs
            
        except Exception as e:
            logger.error(f"Falha no pipeline: {str(e)}")
            raise

class PineconeRagReranker(RagReranker):
    pass

if __name__ == '__main__':
    json_reader = JSONReader(
        levels_back=None,
        is_jsonl=True,
        clean_json=True,
    )
    aux = RagReranker(
        file_extractor={
            ".jsonl": json_reader
        }
    )

    while True:
        user_input = input("Usuário: ")
        response = aux.retrieve_documents(user_input)
        print(f"Chatbot: {response}")
        print(type(response[0]))
    


            #availableReaders = dir(readers)
        #foundReaders = {}
        #for k, v in kwargs.items():
            #foundReaders[k] = v if k in availableReaders else logger.info(f"Reader not acceptable {k}")

            
            #query_engine = self.index.as_query_engine(**kwargs)
            #query_engine = self.index.as_chat_engine(**kwargs)
            #response = query_engine.query(query)
            #response = query_engine.chat(query)
            #print(response)