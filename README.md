# PatentChatBot Project Explanation

## 🎯 What This Project Does
This is a **RAG (Retrieval-Augmented Generation) chatbot** that can answer questions about patent documents. Think of it as a specialized AI assistant that can read your patent files and answer questions about them—citing the exact sources.

---

## 🏗️ Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                           CLIENT (client.py)                          │
│   • Sends patent indexing requests                                   │
│   • Sends questions to the AI                                        │
└───────────────────────────────────┬──────────────────────────────────┘
                                    │ gRPC (Protocol Buffers)
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│                           SERVER (server.py)                          │
│   • Exposes gRPC endpoints                                           │
│   • Handles concurrent requests (ThreadPoolExecutor)                 │
└───────────────────────────────────┬──────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       RAG ENGINE (rag_engine.py)                      │
│                                                                       │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐  │
│   │   Embeddings    │    │   Vector DB     │    │      LLM        │  │
│   │  (HuggingFace)  │◄──►│   (ChromaDB)    │◄──►│  (Llama 3 via   │  │
│   │ all-MiniLM-L6-v2│    │                 │    │     Ollama)     │  │
│   └─────────────────┘    └─────────────────┘    └─────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 🔑 Key Files & What They Do

### 1. `patent_rag.proto` - The Contract (API Definition)
```protobuf
service PatentExpert {
  rpc AskQuestion (QueryRequest) returns (QueryResponse) {}
  rpc IndexPatents (IndexRequest) returns (IndexResponse) {}
}
```
**Why Protocol Buffers?**
- **Strict typing**: Unlike JSON, protobuf enforces types at compile-time
- **Efficient serialization**: Binary format is faster & smaller than JSON
- **Code generation**: Auto-generates Python stubs (`patent_rag_pb2.py`, `patent_rag_pb2_grpc.py`)

**Generated from this command:**
```bash
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. patent_rag.proto
```

---

### 2. `rag_engine.py` - The Brain (RAG Pipeline)

This is where the magic happens. Let's break down the RAG flow:

#### **STEP 1: Indexing (Ingestion Phase)**
```python
def index_documents(self, folder_path):
    # 1. LOAD: Read patent .txt files from folder
    loader = DirectoryLoader(folder_path, glob="**/*.txt", loader_cls=TextLoader)
    docs = loader.load()
    
    # 2. CHUNK: Split long patents into smaller pieces (1000 chars each)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    
    # 3. EMBED & STORE: Convert chunks to vectors, store in ChromaDB
    self.vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=self.embedding_function,  # HuggingFace model
        persist_directory=self.persist_directory
    )
```

**What's happening under the hood:**
- Each chunk of text → converted to a **768-dimensional vector** (embedding)
- These vectors capture the **semantic meaning** of the text
- ChromaDB stores these vectors in `./chroma_db/` (SQLite + binary files)

#### **STEP 2: Querying (RAG in Action)**
```python
def query(self, question, top_k=3):
    # R - RETRIEVAL: Find similar chunks using vector math (cosine similarity)
    results = self.vector_db.similarity_search(question, k=top_k)
    context_text = "\n\n".join([doc.page_content for doc in results])
    
    # A - AUGMENTATION: Inject retrieved context into the prompt
    prompt = f"""
    You are an expert AI Patent Attorney. Use the following context...
    
    Context:
    {context_text}
    
    Question: {question}
    """
    
    # G - GENERATION: LLM generates answer based on augmented prompt
    response = self.llm.invoke(prompt)
```

**Why RAG instead of just asking the LLM?**
- LLMs have knowledge cutoffs (don't know your private patents)
- LLMs can hallucinate facts
- RAG grounds the LLM's response in **your actual documents**
- You get **citations** (source documents)

---

### 3. `server.py` - The gRPC Server

```python
class PatentExpertServicer(patent_rag_pb2_grpc.PatentExpertServicer):
    def __init__(self):
        self.engine = RagEngine()  # Initialize the AI brain
    
    def AskQuestion(self, request, context):
        result = self.engine.query(request.question, top_k=request.top_k or 3)
        return patent_rag_pb2.QueryResponse(
            answer=result["answer"],
            source_documents=result["sources"],
            confidence_score=0.95
        )
```

**Key points:**
- Uses `ThreadPoolExecutor(max_workers=10)` for concurrent requests
- Runs on port `50051` (standard gRPC port)
- Enables **gRPC reflection** (allows tools like `grpcurl` to discover the API)

---

### 4. `client.py` - Example Consumer

```python
with grpc.insecure_channel('localhost:50051') as channel:
    stub = patent_rag_pb2_grpc.PatentExpertStub(channel)
    
    # Step 1: Index patents first
    stub.IndexPatents(patent_rag_pb2.IndexRequest(directory_path="./my_patents"))
    
    # Step 2: Ask questions
    response = stub.AskQuestion(patent_rag_pb2.QueryRequest(
        question="How does the blockchain consensus mechanism optimize for low latency?"
    ))
    print(response.answer)
    print(response.source_documents)  # Citations!
```

---

### 5. `inspect_db.py` - Debugging Tool

This is useful for **peeking inside the vector database**:
- See how many chunks are stored
- View raw document content
- Test vector similarity search without the LLM

---

## 🛠️ Tech Stack Summary

| Component | Technology | Why This Choice |
|-----------|------------|-----------------|
| **API Layer** | gRPC + Protobuf | Type-safe, fast, great for microservices |
| **Orchestration** | LangChain | Simplifies chaining loaders→splitters→embeddings→LLM |
| **Vector DB** | ChromaDB | Simple, embedded, good for prototypes |
| **Embeddings** | `all-MiniLM-L6-v2` | Fast, runs locally, good quality |
| **LLM** | Llama 3 via Ollama | **Runs locally** = privacy (important for patents!) + free |

---

## 🔐 Privacy-First Design

Notice this project runs **everything locally**:
- 🏠 **Ollama** serves Llama 3 on your machine
- 🏠 **HuggingFace embeddings** run locally
- 🏠 **ChromaDB** stores vectors on disk

**No data leaves your computer**—critical when dealing with confidential patent documents.

---

## 📊 The Vector Search Explained

When you ask: *"How does the consensus mechanism optimize latency?"*

1. Your question → embedded to a 768-dim vector: `[0.12, -0.34, 0.87, ...]`
2. ChromaDB compares this vector to all stored patent chunks
3. Uses **cosine similarity** to find the closest matches
4. Returns top-k most similar chunks

This is why the sample patent text:
> *"...proof-of-history algorithm to reduce latency by 40%..."*

Would be retrieved for questions about "latency optimization"—even if the exact words don't match.

---

## 🚀 How to Run This

```bash
# 1. Install dependencies
pip install langchain langchain-community langchain-huggingface chromadb sentence-transformers grpcio grpcio-tools grpcio-reflection

# 2. Install & run Ollama with Llama 3
# Download from https://ollama.ai, then:
ollama pull llama3

# 3. Generate protobuf code (if needed)
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. patent_rag.proto

# 4. Start the server
python server.py

# 5. In another terminal, run the client
python client.py
```

---

## 🎓 Key Takeaways for a Developer

1. **RAG = Retrieval + Augmentation + Generation**
   - Don't just send questions to an LLM
   - First find relevant context, then ask with that context

2. **Embeddings are the magic**
   - Text → Vector allows semantic search
   - "Similar meaning" = "nearby in vector space"

3. **gRPC is great for AI services**
   - Type-safe contracts prevent bugs
   - Efficient binary serialization
   - Easy to add streaming for long responses

4. **Local LLMs are viable**
   - Ollama makes it dead simple
   - Privacy + cost savings + no rate limits


# Technologies Used

- **gRPC**: For high-throughput internal AI microservices with strict contracts (Protobuf) and low latency.
- **LangChain**: For orchestration, abstracted behind a custom service class to avoid vendor lock-in.
- **ChromaDB**: Used as the vector database (production alternatives: Elasticsearch or Milvus).
- **Sentence Transformers**: For text embedding.
- **HuggingFace**: For the QA model.
- **Google Protobuf**: For protocol buffer serialization.
- **Ollama**: For using a local Llama 3 model for privacy (because we are using patent files here) and cost efficiency.
- **Python Standard Libraries**: Such as `os`, `concurrent.futures`, etc.

---

- To generate the code from the interface (patent_rag.proto), execute:
  ```bash
  python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. patent_rag.proto
  ```

- Install dependencies:
  ```bash
  pip install langchain langchain-community langchain-huggingface chromadb sentence-transformers
  ```
