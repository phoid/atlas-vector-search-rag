from pymongo import MongoClient
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.document_loaders import DirectoryLoader
from langchain.llms import OpenAI
import keys

# Set the MongoDB URI, DB, Collection Names

client = MongoClient(keys.MONGO_URI)
dbName = "langchain_demo"
collectionName = "texts"
collection = client[dbName][collectionName]

# Initialize the DirectoryLoader
loader = DirectoryLoader("./sample_files", glob="./*.txt", show_progress=True)
data = loader.load()

# Define the OpenAI Embedding Model we want to use for the source data
# The embedding model is different from the language generation model
embeddings = OpenAIEmbeddings(openai_api_key=keys.OPENAI_KEY)

# Initialize the VectorStore, and
# vectorise the text from the documents using the specified embedding model, and insert them into the specified MongoDB collection
vectorStore = MongoDBAtlasVectorSearch.from_documents(
    data, embeddings, collection=collection
)
