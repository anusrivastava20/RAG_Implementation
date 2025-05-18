import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate


load_dotenv()

rag_prompt_template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question}

Context:
{context}

Answer:
"""

def create_chunks(chunk_size, chunk_overlap=100, is_separator_regex=True):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,is_separator_regex=True )
    print(f"text_splitter: {text_splitter}\n")
    docs = text_splitter.split_documents(documents)
    print(f"docs: {docs}\n")
    return docs

def create_embeddings(openai_key,model,docs):
    embedding = OpenAIEmbeddings(openai_api_type=openai_key, model="text-embedding-3-small")
    print(f"embedding: {embedding}\n")
    vector_store = FAISS.from_documents(docs, embedding)
    print(f"vector_store: {vector_store}")
    return embedding,vector_store


def format_context(retrieved_docs):
    """Format the retrieved documents into a context string."""
    formatted_docs = []
    
    for item in retrieved_docs:
        if isinstance(item, tuple):
            # If item is a tuple (doc, score), get the document
            doc = item[0]
            formatted_docs.append(doc.page_content)
        else:
            # If item is a document object
            formatted_docs.append(item.page_content)
            
    return "\n\n".join(formatted_docs)

def generate_answer(query,retrieved_docs, prompt_template, model_name="gpt-4o-mini", temperature=0):
    """Generate an answer based on the question and retrieved context."""
    llm = ChatOpenAI(model=model_name,temperature=temperature)
    context = format_context(retrieved_docs)
    prompt = PromptTemplate(template=prompt_template, input_variables=['question', 'context'])

    formatted_prompt = prompt.format(question=query, context=context)
    message = [{'role':"user",'content':formatted_prompt}]
    response = llm.invoke(message)
    return response.content

def retrieve_context(query, vector_store, top_k=4, query_vector=None):
    """Retrieve relevant documents for a query.
    
    Args:
        query (str): The query text
        vector_store: The FAISS vector store
        top_k (int): Number of documents to retrieve
        query_vector (list): Pre-computed embedding vector for the query
    """
    # Use pre-computed vector if provided, otherwise use text query
    if query_vector is not None:
        # Use FAISS's similarity_search_by_vector method
        retrieved_docs = vector_store.similarity_search_by_vector(
            query_vector, k=top_k
        )
        # For debugging
        print("Using pre-computed vector for search")
    else:
        # Fall back to the regular text-based search
        retrieved_docs_with_scores = vector_store.similarity_search_with_score(query, k=top_k)
        retrieved_docs = [doc for doc, _ in retrieved_docs_with_scores]
    
    return retrieved_docs

def rag_pipeline(question, vector_store, prompt_template=rag_prompt_template, top_k=4, query_vector=None):
    """Complete RAG pipeline that retrieves context and generates an answer.
    
    Args:
        question (str): The question to answer
        vector_store: The FAISS vector store
        prompt_template (str): The prompt template to use
        top_k (int): Number of documents to retrieve
        query_vector (list): Pre-computed embedding vector for the query
    """
    # Retrieve relevant documents, using pre-computed vector if provided
    retrieved_docs = retrieve_context(question, vector_store, top_k=top_k, query_vector=query_vector)
    
    # Generate answer
    answer = generate_answer(question, retrieved_docs, prompt_template)
    
    # Return both the answer and supporting documents
    return {
        "question": question,
        "answer": answer,
        "context": retrieved_docs
    }
    
def load_document(url):
    loader = UnstructuredWordDocumentLoader(url)
    documents = loader.load()
    return documents


openai_key = os.getenv("OPENAI_API_KEY")
langsmith_tracing= os.getenv("LANGSMITH_TRACING")
langsmith_api_key= os.getenv("LANGSMITH_API_KEY")

model = "text-embedding-3-small"
chunk_size = 1000

chunk_overlap = 200

documents = load_document('./Disney_Digital_Transformation_Report_Complete.docx')

print(f"Number of documents: {len(documents)}\n")

docs = create_chunks(chunk_size,chunk_overlap,True)

embedding, vector_store = create_embeddings(openai_key, model,docs)

sample_text = "How has Disney responded to streaming challenges?"

# Pre-compute the embedding vector for the query
embedded_vector = embedding.embed_query(sample_text)

# Print the dimensions and a few values
print(f"Embedding dimensions: {len(embedded_vector)}")
print(f"First 5 values: {embedded_vector[:5]}")

# Use the pre-computed vector in the RAG pipeline
response = rag_pipeline(sample_text, vector_store, rag_prompt_template, 2, query_vector=embedded_vector)
print(f"Answer: {response['answer']}")