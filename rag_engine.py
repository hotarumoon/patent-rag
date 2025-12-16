import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama  # Using local Llama 3 for privacy/cost


class RagEngine:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        # Use a high-quality open embedding model (runs locally)
        self.embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_db = None
        self.llm = Ollama(model="llama3")  # Assumes Ollama is running locally

    def index_documents(self, folder_path):
        """Ingests patents from a folder, chunks them, and embeds them."""
        print(f"Loading documents from {folder_path}...")
        loader = DirectoryLoader(folder_path, glob="**/*.txt", loader_cls=TextLoader)
        docs = loader.load()

        # Split text into chunks (Patents are long, we need chunks)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)

        # Store in Vector DB
        self.vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding_function,
            persist_directory=self.persist_directory
        )
        return len(chunks)

    def query(self, question, top_k=3):
        """Retrieves context and generates an answer."""
        if not self.vector_db:
            # Load existing DB if not already loaded in memory
            self.vector_db = Chroma(persist_directory=self.persist_directory,
                                    embedding_function=self.embedding_function)

        # 1. Retrieve relevant chunks
        results = self.vector_db.similarity_search(question, k=top_k)
        context_text = "\n\n".join([doc.page_content for doc in results])
        sources = [doc.metadata.get('source', 'Unknown') for doc in results]

        # 2. Augment the prompt (The "A" in RAG)
        prompt = f"""
        You are an expert AI Patent Attorney. Use the following context from my patent portfolio to answer the question. 
        If the answer is not in the context, say "I don't find that in your patents."

        Context:
        {context_text}

        Question: {question}

        Answer:
        """

        # 3. Generate (The "G" in RAG)
        response = self.llm.invoke(prompt)

        return {
            "answer": response,
            "sources": list(set(sources))  # Unique sources
        }