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
