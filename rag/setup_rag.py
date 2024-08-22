from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as gemini
import os
import json

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Create a Pinecone index
pc.create_index(
    name="rag",
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)

# Load the review data
data = json.load(open("reviews.json"))

processed_data = []

# Initialize Google Gemini
gemini.configure(api_key=os.getenv("GOOGLE_GEMINI_API_KEY"))

# Create embeddings for each review using Google Gemini
for review in data["reviews"]:
    # Embed the review text
    response = gemini.generate_embeddings(model="textembedding-gecko-001", texts=[review['review']])
    embedding = response['data'][0]['embedding']

    processed_data.append(
        {
            "values": embedding,
            "id": review["professor"],
            "metadata": {
                "review": review["review"],
                "subject": review["subject"],
                "stars": review["stars"],
            }
        }
    )

# Insert the embeddings into the Pinecone index
index = pc.Index("rag")
upsert_response = index.upsert(
    vectors=processed_data,
    namespace="ns1",
)

print(f"Upserted count: {upsert_response['upserted_count']}")

# Print index statistics
print(index.describe_index_stats())
