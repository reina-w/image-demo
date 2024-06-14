import os
import urllib.request
import openai
from openai import OpenAI
from pymilvus import MilvusClient
from tqdm import tqdm
import streamlit as st
import ssl
import certifi
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Logging is configured and working.")

st.set_page_config(layout="wide")

# Logo
st.image("Milvus Logo_Official.png", width=200)

# Set Streamlit title
st.title("Milvus and OpenAI Embedding Search")

# Set OpenAI API key
# openai.api_key = st.secrets["api_key"]
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Set SSL context
ssl_context = ssl.create_default_context(cafile=certifi.where())

# URL for the file to be retrieved
url = "https://raw.githubusercontent.com/milvus-io/milvus/master/DEVELOPMENT.md"
file_path = "./Milvus_DEVELOPMENT.md"

# Retrieve the URL content
with urllib.request.urlopen(url, context=ssl_context) as response:
    with open(file_path, "wb") as file:
        file.write(response.read())

# Read the downloaded file
with open(file_path, "r") as file:
    file_text = file.read()

# Split text into lines
text_lines = file_text.split("# ")

openai_client = OpenAI()

# Cache for embeddings
@st.cache_resource
def get_embedding_cache():
    return {}

embedding_cache = get_embedding_cache()

# Define a function to get text embeddings
def emb_text(text):
    if text in embedding_cache:
        return embedding_cache[text]
    else:
        embedding = (
            openai_client.embeddings.create(input=text, model="text-embedding-3-small")
            .data[0]
            .embedding
        )
        embedding_cache[text] = embedding
        return embedding

# Get a test embedding to determine the embedding dimension
test_embedding = emb_text("This is a test")
embedding_dim = len(test_embedding)

# Initialize Milvus client
@st.cache_resource
def get_milvus_client(uri):
    logger.info("Setting up Milvus client...")
    return MilvusClient(uri=uri)

milvus_client = get_milvus_client(uri="./milvus_demo.db")

# milvus_client = MilvusClient("./milvus_demo.db")
collection_name = "my_rag_collection"

# Drop collection if it exists
if milvus_client.has_collection(collection_name):
    milvus_client.drop_collection(collection_name)

# Check if the collection exists
logger.info(f"Database file exists: {os.path.exists('milvus_demo.db')}")
if not milvus_client.has_collection(collection_name):
    milvus_client.create_collection(
        collection_name=collection_name,
        dimension=embedding_dim,
        metric_type="IP",  # Inner product distance
        consistency_level="Strong",  # Strong consistency level
    )

# # Create a new collection
# milvus_client.create_collection(
#     collection_name=collection_name,
#     dimension=embedding_dim,
#     metric_type="IP",  # Inner product distance
#     consistency_level="Strong",  # Strong consistency level
# )

# Prepare data for insertion into Milvus
data = []

for i, line in enumerate(tqdm(text_lines, desc="Creating embeddings")):
    data.append({"id": i, "vector": emb_text(line), "text": line})

# Insert data into Milvus collection
milvus_client.insert(collection_name=collection_name, data=data)

col1, col2 = st.columns([1, 1])

retrieved_lines_with_distances = []

# Initialize chat history in session state (optional: to save chat history)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Add content to each column
with col1:
    with st.form("my_form"):
        question = st.text_area("Enter your question:")

        # what is the hardware requirements specification if I want to build Milvus and run from source code?
        submitted = st.form_submit_button("Submit")

        if question and submitted:
            # Search in Milvus collection
            search_res = milvus_client.search(
                collection_name=collection_name,
                data=[emb_text(question)],  # Convert question to embedding vector
                limit=3,  # Return top 3 results
                search_params={"metric_type": "IP", "params": {}},  # Inner product distance
                output_fields=["text"],  # Return the text field
            )

            # Retrieve lines and distances
            retrieved_lines_with_distances = [
                (res["entity"]["text"], res["distance"]) for res in search_res[0]
            ]

            # Create context from retrieved lines
            context = "\n".join(
                [line_with_distance[0] for line_with_distance in retrieved_lines_with_distances]
            )

            # Define system and user prompts
            SYSTEM_PROMPT = """
            Human: You are an AI assistant. You are able to find answers to the questions from the contextual passage snippets provided.
            """
            USER_PROMPT = f"""
            Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
            <context>
            {context}
            </context>
            <question>
            {question}
            </question>
            """

            # Generate response using OpenAI's GPT-3.5-turbo model
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT},
                ],
            )

            answer = response.choices[0].message.content

            # Update chat history (optional: to save chat history)
            st.session_state.chat_history.append({"role": "user", "content": question})
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

            # Display the question and response in a chatbot-style box
            st.write("Chatbot Conversation:")
            # st.chat_message("user").write(question)
            # st.chat_message("assistant").write(answer)

            for chat in st.session_state.chat_history: # (optional: to save chat history)
                st.chat_message(chat["role"]).write(chat["content"])

        
with col2:
    # Display the retrieved lines in a more readable format
    st.subheader("Retrieved Lines with Distances:")
    for idx, (line, distance) in enumerate(retrieved_lines_with_distances, 1):
        st.markdown("---")
        st.markdown(f"**Result {idx}:**")
        st.markdown(f"> {line}")
        st.markdown(f"*Distance: {distance:.2f}*")