import grpc
import patent_rag_pb2
import patent_rag_pb2_grpc


def run():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = patent_rag_pb2_grpc.PatentExpertStub(channel)

        # 1. First, Index the Patents (Put her dummy patent text files in ./my_patents)
        print("Indexing Patents...")
        index_response = stub.IndexPatents(patent_rag_pb2.IndexRequest(directory_path="./my_patents"))
        print(f"Indexing Status: {index_response.message}\n")

        # 2. Ask a question about her specific work
        question = "How does the blockchain consensus mechanism described in the patent optimize for low latency?"
        print(f"Asking: {question}...")

        response = stub.AskQuestion(patent_rag_pb2.QueryRequest(question=question))

        print("-" * 30)
        print(f"🤖 AI Answer: {response.answer}")
        print(f"📄 Sources: {response.source_documents}")
        print("-" * 30)


if __name__ == '__main__':
    run()