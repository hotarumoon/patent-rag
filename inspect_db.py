from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Setup the Connection (Must use same embedding model as ingestion!)
print(" Connecting to the Database...")
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)

# 2. Check the Count
count = db._collection.count()
print(f" Total Document Chunks stored: {count}")

# 3. Peek at the Raw Data (The first 3 chunks)
if count > 0:
    print("\n Peeking at the first 3 documents:")
    # Chroma's internal method to fetch data without vectors
    data = db._collection.get(limit=3)

    ids = data['ids']
    metadatas = data['metadatas']
    documents = data['documents']

    for i in range(len(ids)):
        print(f"-" * 40)
        print(f"ID: {ids[i]}")
        print(f"Source: {metadatas[i]}")
        print(f"Content (Preview): {documents[i][:200]}...")  # Show first 200 chars
else:
    print(" The database is empty! Did you run the Indexing step in client.py?")

# 4. Test a 'Raw' Similarity Search (No LLM, just Math)
print("\n" + "=" * 40)
print(" TESTING VECTOR MATH")
query = "consensus mechanism"  # Change this to search for whatever you want
print(f"Searching for nearest neighbors to: '{query}'")

results = db.similarity_search(query, k=2)

for i, doc in enumerate(results):
    print(f"\nMatch #{i + 1}:")
    print(f"Content: {doc.page_content[:150]}...")
    print(f"Source: {doc.metadata}")