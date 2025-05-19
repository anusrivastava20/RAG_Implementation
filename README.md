# RAG Applications

This repository contains two Retrieval-Augmented Generation (RAG) implementations:
1. A simple RAG application (`simple_rag.py`)
2. An advanced RAG application (`advance_rag.py`)

Both applications demonstrate how to create a question-answering system using document retrieval and LLMs.

## Table of Contents
- [Simple RAG Application](#simple-rag-application)
- [Advanced RAG Application](#advanced-rag-application)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Customization](#customization)

## Simple RAG Application

### Features
- Document loading and chunking
- Text embeddings with OpenAI
- FAISS vector store for efficient similarity search
- Pre-computed vector embeddings for query optimization
- Simple question-answering pipeline

### Implementation Details
The simple RAG application follows these steps:
1. Loads documents using UnstructuredWordDocumentLoader
2. Splits documents into manageable chunks using RecursiveCharacterTextSplitter
3. Creates embeddings for document chunks and stores them in a FAISS vector database
4. Retrieves relevant document chunks using vector similarity search
5. Generates answers using an LLM (ChatOpenAI) with a custom prompt template

### Key Components
- `create_chunks()`: Splits documents into smaller chunks
- `create_embeddings()`: Creates vector embeddings for document chunks
- `retrieve_context()`: Retrieves relevant documents based on query similarity
- `generate_answer()`: Generates answers based on retrieved context
- `rag_pipeline()`: Orchestrates the entire RAG process

## Advanced RAG Application

### Features
- All features from the simple RAG application
- OOP design with `AdvancedRAG` class
- Conversation history management
- Query rewriting based on conversation context
- Hybrid search combining vector and keyword-based search
- Self-evaluation of document relevance
- Logging for performance analysis
- Vector store persistence (save/load)

### Implementation Details
The advanced RAG application enhances the simple version with these additional steps:
1. Maintains conversation history for contextual understanding
2. Rewrites user queries to improve retrieval effectiveness
3. Implements hybrid search combining vector similarity with keyword matching
4. Evaluates relevance of retrieved documents
5. Logs interactions for analysis and improvement
6. Supports persistence of the vector store

### Key Components
- `rewrite_query()`: Improves queries based on conversation context
- `hybrid_search()`: Combines vector and keyword-based search
- `evaluate_relevance()`: Assesses document relevance to query
- `advanced_query()`: Main pipeline with all advanced features
- `save_vector_store()`/`load_vector_store()`: Persistence methods
- Conversation management with `add_to_history()`/`get_conversation_history()`

## Prerequisites
- Python 3.8+
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rag-applications.git
cd rag-applications
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

### Simple RAG Application

```bash
python simple_rag.py
```

To customize the application:
```python
# Load your own document
documents = load_document('./path/to/your/document.docx')

# Create vector store
docs = create_chunks(1000, 200, True)
embedding, vector_store = create_embeddings(openai_key, model, docs)

# Ask a question
response = rag_pipeline("Your question here?", vector_store)
print(f"Answer: {response['answer']}")
```

### Advanced RAG Application

```bash
python advance_rag.py
```

To customize the advanced application:
```python
# Initialize the RAG system
rag = AdvancedRAG()

# Load documents and create vector store
documents = rag.load_document('./path/to/your/document.docx')
chunks = rag.create_chunks(documents)
rag.create_vector_store(chunks)

# Create a conversation ID for persistent history
conversation_id = "user_123"

# Ask a question
result = rag.advanced_query("Your question here?", conversation_id)
print(f"Answer: {result['answer']}")

# Save the vector store for future use
rag.save_vector_store("./your_vector_store")
```

## Customization

### Changing Chunk Size
Modify the chunk size and overlap to balance between context granularity and relevance:
```python
# Simple RAG
docs = create_chunks(chunk_size=500, chunk_overlap=100)

# Advanced RAG
chunks = rag.create_chunks(documents, chunk_size=500, chunk_overlap=100)
```

### Using Different Embedding Models
Change the embedding model by modifying the model parameter:
```python
# Simple RAG
embedding = OpenAIEmbeddings(model="text-embedding-ada-002")

# Advanced RAG
rag = AdvancedRAG(model="text-embedding-ada-002")
```

### Custom Prompt Templates
Modify the prompt template to change how the LLM generates answers:
```python
custom_prompt = """
[Your custom prompt here with {question} and {context} variables]
"""

# Simple RAG
response = rag_pipeline(question, vector_store, prompt_template=custom_prompt)

# Advanced RAG
# Modify the advanced_rag_prompt variable at the top of the file
```
