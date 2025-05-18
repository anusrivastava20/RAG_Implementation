import os
import json
import datetime
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate

load_dotenv()

# Advanced RAG prompt with reflection component
advanced_rag_prompt = """
You are an advanced question-answering assistant. Use the following retrieved context to answer the question. 
Consider the relevance and reliability of each piece of context in formulating your answer.

Question: {question}

Context:
{context}

First, assess the relevance of the provided context to the question on a scale of 1-10: [relevance assessment]

Then provide your answer, using only information from the context. If the context doesn't contain sufficient information, acknowledge the limitations in your answer.
Keep your answer concise (maximum 3 sentences unless more detail is absolutely necessary).

Answer:
"""

class AdvancedRAG:
    def __init__(self, model="text-embedding-3-small", llm_model="gpt-4o-mini"):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.embedding_model = OpenAIEmbeddings(model=model)
        self.llm = ChatOpenAI(model=llm_model, temperature=0)
        self.vector_store = None
        self.conversation_history = {}
        
    def load_document(self, file_path):
        """Load document from file path."""
        loader = UnstructuredWordDocumentLoader(file_path)
        documents = loader.load()
        print(f"Loaded {len(documents)} documents")
        return documents
    
    def create_chunks(self, documents, chunk_size=1000, chunk_overlap=200):
        """Split documents into chunks."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
            is_separator_regex=True
        )
        docs = text_splitter.split_documents(documents)
        print(f"Created {len(docs)} chunks")
        return docs
    
    def create_vector_store(self, docs):
        """Create and populate vector store with document chunks."""
        self.vector_store = FAISS.from_documents(docs, self.embedding_model)
        print("Vector store created successfully")
        return self.vector_store
    
    def rewrite_query(self, query, conversation_id=None):
        """Rewrite query to improve retrieval results."""
        # Consider conversation history if available
        history = self.get_conversation_history(conversation_id)
        history_text = ""
        
        if history:
            history_text = "\n".join([f"User: {h['question']}\nAssistant: {h['answer']}" 
                                    for h in history[-3:]])  # Use last 3 exchanges
        
        rewrite_prompt = f"""
        You are an AI assistant helping to improve search queries. Rewrite the following query to make it more effective for retrieving relevant information.
        Make it more specific and include key terms that would help in document retrieval.
        
        {f'Recent conversation history:\n{history_text}\n\n' if history_text else ''}
        Original query: {query}
        
        Rewritten query:
        """
        
        messages = [{"role": "user", "content": rewrite_prompt}]
        response = self.llm.invoke(messages)
        rewritten_query = response.content.strip()
        
        print(f"Original query: {query}")
        print(f"Rewritten query: {rewritten_query}")
        
        return rewritten_query
    
    def hybrid_search(self, query, top_k=4, alpha=0.7):
        """Implement hybrid search combining vector search with keyword relevance."""
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Call create_vector_store first.")
        
        # Get vector search results
        vector_results = self.vector_store.similarity_search_with_score(query, k=top_k*2)
        
        # Simple keyword matching score (as a basic alternative to BM25)
        def keyword_score(doc, query_terms):
            content = doc.page_content.lower()
            return sum(2 if term in content else 0 for term in query_terms)
        
        # Get query terms (simple tokenization)
        query_terms = set(query.lower().split())
        
        # Calculate hybrid scores
        hybrid_results = []
        for doc, vector_score in vector_results:
            # Convert similarity score (higher is better)
            normalized_vector_score = 1.0 / (1.0 + vector_score)
            
            # Get keyword score
            kw_score = keyword_score(doc, query_terms)
            normalized_kw_score = kw_score / (len(query_terms) * 2)  # Normalize to 0-1
            
            # Calculate hybrid score (weighted combination)
            hybrid_score = (alpha * normalized_vector_score) + ((1-alpha) * normalized_kw_score)
            
            hybrid_results.append((doc, hybrid_score))
        
        # Sort by hybrid score (descending) and take top_k
        hybrid_results.sort(key=lambda x: x[1], reverse=True)
        return hybrid_results[:top_k]
    
    def format_context(self, retrieved_docs):
        """Format retrieved documents into a context string."""
        formatted_docs = []
        
        for item in retrieved_docs:
            if isinstance(item, tuple):
                doc, score = item
                formatted_docs.append(f"[Relevance: {score:.2f}]\n{doc.page_content}")
            else:
                formatted_docs.append(item.page_content)
                
        return "\n\n---\n\n".join(formatted_docs)
    
    def generate_answer(self, query, retrieved_docs, prompt_template):
        """Generate an answer based on query and retrieved documents."""
        context = self.format_context(retrieved_docs)
        
        prompt = PromptTemplate(template=prompt_template, input_variables=['question', 'context'])
        formatted_prompt = prompt.format(question=query, context=context)
        
        messages = [{'role': "user", 'content': formatted_prompt}]
        response = self.llm.invoke(messages)
        
        return response.content
    
    def evaluate_relevance(self, query, retrieved_docs):
        """Evaluate relevance of retrieved documents."""
        eval_prompt = f"""
        Evaluate the relevance of these documents to the query: "{query}"
        
        Rate the overall relevance on a scale of 1-10, where 10 is perfectly relevant.
        Provide a single number as your answer.
        """
        
        context = self.format_context(retrieved_docs)
        messages = [{'role': "user", 'content': f"{eval_prompt}\n\nDocuments:\n{context}"}]
        response = self.llm.invoke(messages)
        
        try:
            relevance_score = float(response.content.strip())
            return relevance_score
        except:
            return 5.0  # Default score if parsing fails
    
    def get_conversation_history(self, conversation_id):
        """Get conversation history for a given ID."""
        if not conversation_id:
            return []
        return self.conversation_history.get(conversation_id, [])
    
    def add_to_history(self, conversation_id, question, answer, context=None):
        """Add interaction to conversation history."""
        if not conversation_id:
            return
            
        if conversation_id not in self.conversation_history:
            self.conversation_history[conversation_id] = []
            
        self.conversation_history[conversation_id].append({
            "timestamp": datetime.datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "context": context[:3] if context else None  # Store only first 3 docs to save space
        })
        
        # Limit history size
        if len(self.conversation_history[conversation_id]) > 10:
            self.conversation_history[conversation_id] = self.conversation_history[conversation_id][-10:]
    
    def log_interaction(self, question, answer, retrieved_docs, metrics=None):
        """Log interactions for evaluation purposes."""
        timestamp = datetime.datetime.now().isoformat()
        
        log_entry = {
            "timestamp": timestamp,
            "question": question,
            "answer": answer,
            "context": [doc.page_content if not isinstance(doc, tuple) else doc[0].page_content 
                        for doc in retrieved_docs[:3]],  # First 3 docs only
            "metrics": metrics or {}
        }
        
        os.makedirs("logs", exist_ok=True)
        with open("logs/rag_interactions.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def advanced_query(self, query, conversation_id=None, top_k=4):
        """Advanced RAG pipeline with conversation memory, query rewriting, and evaluation."""
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
            
        # Get conversation history
        history = self.get_conversation_history(conversation_id)
        
        # Rewrite query considering conversation context
        if history:
            rewritten_query = self.rewrite_query(query, conversation_id)
        else:
            rewritten_query = query
            
        # Retrieve relevant documents using hybrid search
        retrieved_docs = self.hybrid_search(rewritten_query, top_k=top_k)
        
        # Evaluate document relevance
        relevance_score = self.evaluate_relevance(query, retrieved_docs)
        
        # Generate answer
        answer = self.generate_answer(query, retrieved_docs, advanced_rag_prompt)
        
        # Add to conversation history
        self.add_to_history(conversation_id, query, answer, retrieved_docs)
        
        # Log interaction
        self.log_interaction(query, answer, retrieved_docs, {
            "relevance_score": relevance_score,
            "rewritten_query": rewritten_query
        })
        
        # Return comprehensive result
        return {
            "question": query,
            "rewritten_query": rewritten_query,
            "answer": answer,
            "context": [doc for doc, _ in retrieved_docs],
            "relevance_score": relevance_score
        }
        
    def save_vector_store(self, path):
        """Save the vector store to disk."""
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        self.vector_store.save_local(path)
        print(f"Vector store saved to {path}")
        
    def load_vector_store(self, path):
        """Load the vector store from disk."""
        self.vector_store = FAISS.load_local(path, self.embedding_model)
        print(f"Vector store loaded from {path}")
        return self.vector_store

# Example usage
def main():
    rag = AdvancedRAG()
    
    # Option 1: Create new vector store from documents
    documents = rag.load_document('./Disney_Digital_Transformation_Report_Complete.docx')
    chunks = rag.create_chunks(documents)
    rag.create_vector_store(chunks)
    
    # Option 2: Load existing vector store
    # rag.load_vector_store("./disney_vector_store")
    
    # Create a conversation ID for persistent history
    conversation_id = "user_123"
    
    # Process a query with the advanced RAG pipeline
    query = "How has Disney responded to streaming challenges?"
    result = rag.advanced_query(query, conversation_id)
    
    print("\n=== Results ===")
    print(f"Question: {result['question']}")
    print(f"Rewritten query: {result['rewritten_query']}")
    print(f"Relevance score: {result['relevance_score']}/10")
    print(f"Answer: {result['answer']}")
    
    # Save vector store for future use
    rag.save_vector_store("./disney_vector_store")

if __name__ == "__main__":
    main()