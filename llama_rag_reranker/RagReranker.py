from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, QueryBundle, Document
import logging
from llama_index.core.postprocessor import LLMRerank
from llama_index.readers.json import JSONReader
from llama_index.core.retrievers import VectorIndexRetriever
from utils import get_valid_kwargs
from .StructuredOutput import StructuredOutput
from utils import RELEVANT_TEMPLATE
from pydantic import BaseModel, Field
from llama_index.core.schema import NodeWithScore


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

    def __init__(self, data_dir: str = "docs", file_extractor: dict = None, **kwargs) -> None:
        self.data_dir = data_dir
        self.documents: list[Document]
        self.index: VectorStoreIndex
        self.retriever: VectorIndexRetriever
        self.reranker: LLMRerank
        self.relevancy: StructuredOutput = StructuredOutput()

        self._reader(file_extractor=file_extractor, **kwargs)
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

    def _relevancy(self, query, retrieved_nodes) -> list[NodeWithScore]:
        relevant = []
        for node in retrieved_nodes:
            aux = node.node.get_text()
            query = RELEVANT_TEMPLATE.format(context_str=aux, query_str=query)
            if self.relevancy.query_output(query).isRelevant:
                relevant.append(node)
        return relevant
    
    def retrieve_documents(self, query: str, **kwargs) -> list[NodeWithScore]:
        try:
            query_bundle = QueryBundle(query)
            retrieved_nodes = self.retriever.retrieve(query_bundle)
            retrieved_nodes = self.reranker.postprocess_nodes(
                retrieved_nodes, query_bundle
            )
            relevant_documents = self._relevancy(query, retrieved_nodes)
            #documents = [node.node.get_text() for node in retrieved_nodes]
            #logger.info(f"Successfully retrieved: {query}")
            #return documents
            return relevant_documents
        except Exception as e:
            logger.error(f"Error during retrieving: {str(e)}")
            raise
'''
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
'''

#Extrair informações relevantes do paciente e salvar
#result.dict()["isRelevant"]