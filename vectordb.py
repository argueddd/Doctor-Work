from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

client = MilvusClient(uri="http://localhost:19530")
model = SentenceTransformer('DMetaSoul/Dmeta-embedding')

# if client.has_collection(collection_name="demo_collection"):
#     client.drop_collection(collection_name="demo_collection")
#
# client.create_collection(
#     collection_name="demo_collection",
#     dimension=768,  # The vectors we will use in this demo has 768 dimensions
# )

collection = client.get_load_state(
    collection_name="demo_collection"
)

docs = [
    "Artificial intelligence was founded as an academic discipline in 1956.",
    "Alan Turing was the first person to conduct substantial research in AI.",
    "Born in Maida Vale, London, Turing was raised in southern England.",
]

vectors = model.encode(docs, normalize_embeddings=True)

data = [
    {"id": i, "vector": vectors[i], "text": docs[i], "subject": "history"}
    for i in range(len(vectors))
]

query_vectors = model.encode(["who is Alan Turing"])

# insert_result = client.upsert(collection_name="demo_collection", data=data)
# print(insert_result)

res = client.search(
    collection_name="demo_collection",  # target collection
    data=query_vectors,  # query vectors
    limit=2,  # number of returned entities
    output_fields=["text"],  # specifies fields to be returned
)

print(res)
