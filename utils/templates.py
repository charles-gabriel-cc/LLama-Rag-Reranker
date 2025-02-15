from llama_index.core import PromptTemplate

CITATION_QA_TEMPLATE = PromptTemplate(
    "Please provide an answer based solely on the provided sources. "
    "When referencing information from a source, "
    "cite the appropriate source(s) using their corresponding numbers. "
    "Every answer should include at least one source citation. "
    "Only cite a source when you are explicitly referencing it. "
    "If none of the sources are helpful, you should indicate that. "
    "For example:\n"
    "Source 1:\n"
    "The sky is red in the evening and blue in the morning.\n"
    "Source 2:\n"
    "Water is wet when the sky is red.\n"
    "Query: When is water wet?\n"
    "Answer: Water will be wet when the sky is red [2], "
    "which occurs in the evening [1].\n"
    "Now it's your turn. Below are several numbered sources of information:"
    "\n------\n"
    "{context_str}"
    "\n------\n"
    "Query: {query_str}\n"
    "Answer: "
)

RELEVANT_TEMPLATE = PromptTemplate(
        "You are an intermediate layer of an llm to classify whether this material returned by RAG is really relevant to answer the user's question, True for relevant, and False for irrelevant\n"
        "Materials:\n"
        "{context_str}\n"
        "Query: {query_str}\n"
)

RERANKER_TEMPLATE = PromptTemplate(
    "teste"
)

CONTEXTUAL_QUERY_STR = PromptTemplate(
    ""
)

NEED_RAG_TEMPLATE = PromptTemplate(
    "You are an assistant responsible for determining whether retrieval-augmented generation (RAG) "
    "should be used to enhance the context of a user query.\n\n"
    "User query: \"{query_str}\"\n\n"
    "If the query requires external knowledge or factual information beyond the available context, reply with 'True'. "
    "Otherwise, reply with 'False'.\n\n"
    "Answer only with 'True' or 'False'."
)

