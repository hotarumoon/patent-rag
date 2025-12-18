import grpc
from concurrent import futures
import time

import warnings
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality isn't compatible")

# Import the AI Logic
from rag_engine import RagEngine

# Import generated protobuf code
import patent_rag_pb2
import patent_rag_pb2_grpc
from grpc_reflection.v1alpha import reflection

class PatentExpertServicer(patent_rag_pb2_grpc.PatentExpertServicer):
    def __init__(self):
        self.engine = RagEngine()
        # Pre-load DB check could go here

    def IndexPatents(self, request, context):
        try:
            count = self.engine.index_documents(request.directory_path)
            return patent_rag_pb2.IndexResponse(
                success=True,
                documents_processed=count,
                message="Indexing completed successfully."
            )
        except Exception as e:
            return patent_rag_pb2.IndexResponse(success=False, message=str(e))

    def AskQuestion(self, request, context):
        print(f"Received Question: {request.question}")
        start_time = time.time()

        # Call the logic
        result = self.engine.query(request.question, top_k=request.top_k or 3)

        # Map to Proto response
        return patent_rag_pb2.QueryResponse(
            answer=result["answer"],
            source_documents=result["sources"],
            confidence_score=0.95  # Placeholder for actual confidence logic
        )


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    patent_rag_pb2_grpc.add_PatentExpertServicer_to_server(PatentExpertServicer(), server)

    # Enable reflection
    SERVICE_NAMES = (
        patent_rag_pb2.DESCRIPTOR.services_by_name['PatentExpert'].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)

    port = '[::]:50051'
    server.add_insecure_port(port)
    print(f"AI Patent Server started on {port}")
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    serve()