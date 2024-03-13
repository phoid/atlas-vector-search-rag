from pymongo import MongoClient
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores.mongodb_atlas import MongoDBAtlasVectorSearch
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.llms import OpenAI
import keys

# Set the MongoDB URI, DB, Collection Names

client = MongoClient(keys.MONGO_URI)
dbName = "langchain_demo"
collectionName = "texts"
collection = client[dbName][collectionName]

# Initialize the DirectoryLoader
loader = DirectoryLoader("./samples", glob="./*.txt", show_progress=True)
data = loader.load()

# Define the OpenAI Embedding Model we want to use for the source data
# The embedding model is different from the language generation model
embeddings = OpenAIEmbeddings(openai_api_key=keys.OPENAI_KEY)

# Initialize the VectorStore, and
# vectorise the text from the documents using the specified embedding model, and insert them into the specified MongoDB collection
vectorStore = MongoDBAtlasVectorSearch.from_documents(
    data, embeddings, collection=collection
)
atlasSI = {
    "fields": [
        {
            "path": "embedding",
            "numDimensions": 1536,
            "similarity": "cosine",
            "type": "vector",
        }
    ]
}
client.get_database(dbName).get_collection(collectionName).create_index(atlasSI)
