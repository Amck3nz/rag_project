
import os
import requests
import pymongo
#from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()

mdb_uri= os.environ.get('MONGO_DB_CONNECT_STRING')

hf_token= os.environ.get('HUGGING_FACE_TOKEN')
embedding_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"


# Connect to Mondo DB
client = pymongo.MongoClient(mdb_uri)
db = client.sample_mflix
collection = db.movies



# Create/store movie plot data field embeddings
def generate_embedding(text:str) -> list[float]:
    response = requests.post(embedding_url,
                             headers={"Authorization": f"Bearer {hf_token}"},
                             json={"inputs": text}
                             )

    if response.status_code != 200:
        raise ValueError(f"Request failed with status code {response.status_code}: {response.text}")
    
    return response.json()



#for entry in collection.find({'plot':{'$exists': True}}).limit(50):      # For documents in our collection, find items where the plot exists (just 50 b/c rate limits & time)
#    entry['plot_embedding_hf'] = generate_embedding(entry['plot'])      # Add new plot embedding field to db
#    collection.replace_one({'_id' : entry['_id']}, entry)               # Update info for that movie with the same info PLUS new embedding field (alternatively could creat new seperate collection for embeddings)
#

#Vector Search (using mongo db collection vector search aggregation pipeline)
query = "found footage horror movie where people explore a scary location"

results = collection.aggregate([
    {"$vectorSearch": {
        "queryVector": generate_embedding(query),
        "path": "plot_embedding_hf",
        "numCandidates":100,
        "limit":3,
        "index": "PlotSemanticSearch",
    }}
]);

for document in results:
    print(f"Movie Rec: {document['title']} | Plot: {document['plot']}\n")


#print(generate_embedding("steak is the best food"))