# Done:
# added the slider feature
# moved milvus logo to sidebar
# added the cropper feature

import streamlit as st
from streamlit_cropper import st_cropper
import streamlit_cropper 
import os
import requests
import zipfile
import torch
from PIL import Image
import timm
from sklearn.preprocessing import normalize
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from pymilvus import MilvusClient
import certifi
import logging
import threading

def _recommended_box2(img: Image, aspect_ratio: tuple = None) -> dict:
    width, height = img.size
    return {'left': int(0), 'top': int(0), 'width': int(width-2), 'height': int(height-2)}
streamlit_cropper._recommended_box = _recommended_box2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Logging is configured and working.")

# Streamlit UI
st.set_page_config(layout="wide")

# Create path to the model file (may subject to change depending on where you store the file)
documents_dir = os.path.expanduser("~/Documents")
folder_name = "streamlit_app_image"
MODEL_PATH = os.path.join(documents_dir, folder_name, "feature_extractor_model.pth")
# Ensure the directory exists
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

def download_file(url, dest):
    response = requests.get(url, verify=certifi.where())
    with open(dest, 'wb') as f:
        f.write(response.content)

# Download and unzip data if not already done
zip_path = 'reverse_image_search.zip'
if not os.path.exists(zip_path):
    url = 'https://github.com/milvus-io/pymilvus-assets/releases/download/imagedata/reverse_image_search.zip'
    download_file(url, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('.')

class FeatureExtractor:
    def __init__(self, modelname):
        self.model = timm.create_model(modelname, pretrained=True, num_classes=0, global_pool="avg")
        self.model.eval()
        self.input_size = self.model.default_cfg["input_size"]
        config = resolve_data_config({}, model=modelname)
        self.preprocess = create_transform(**config)
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)
    
    @staticmethod
    def load(path, modelname):
        model = FeatureExtractor(modelname)
        model.model.load_state_dict(torch.load(path))
        model.model.eval()
        return model

    def __call__(self, input):
        input_image = input.convert("RGB")
        input_image = self.preprocess(input_image)
        input_tensor = input_image.unsqueeze(0)
        with torch.no_grad():
            output = self.model(input_tensor)
        feature_vector = output.squeeze().numpy()
        return normalize(feature_vector.reshape(1, -1), norm="l2").flatten()


@st.cache_resource
def load_model(modelname, path=MODEL_PATH):
    logger.info("Loading model...")
    if os.path.exists(path):
        logger.info("Model loaded from cache.")
        return FeatureExtractor.load(path, modelname)
    else:
        model = FeatureExtractor(modelname)
        model.save(path)
        logger.info("Model initialized and saved.")
        return model

@st.cache_resource
def get_milvus_client(uri):
    logger.info("Setting up Milvus client")
    return MilvusClient(uri=uri)

extractor = load_model("resnet34")

root = "./train"

extractor = load_model("resnet34")
def insert_embeddings(client):
    print('inserting')
    global extractor
    root = "./train"
    for dirpath, foldername, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(".JPEG"):
                filepath = os.path.join(dirpath, filename)
                img = Image.open(filepath)
                image_embedding = extractor(img)
                client.insert(
                    "image_embeddings",
                    {"vector": image_embedding, "filename": filepath},
                )
@st.cache_resource
def db_exists_check():
    logger.info(f"Database file exists: {os.path.exists('example.db')}")
    if not os.path.exists("example.db"):
        client = get_milvus_client(uri="example.db")
        client.create_collection(
            collection_name="image_embeddings",
            vector_field_name="vector",
            dimension=512,
            auto_id=True,
            enable_dynamic_field=True,
            metric_type="COSINE",
        )
        insert_embeddings(client)
    
    else:
        client = get_milvus_client(uri="example.db")
    return client

client = db_exists_check()




# Logo
st.sidebar.image("Milvus Logo_Official.png", width=200)

st.title("Image Similarity Search :frame_with_picture: ")

query_image = "temp.jpg"
cols = st.columns(5)

uploaded_file = st.sidebar.file_uploader("Choose an image...", type="jpeg")

if uploaded_file is not None:
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    # st.sidebar.image(query_image, caption="Query Image", use_column_width=True)
    # cropper
    # Get a cropped image from the frontend
    uploaded_img = Image.open(uploaded_file)
    width, height = uploaded_img.size

    new_width = 370
    new_height = int((new_width / width) * height)
    uploaded_img = uploaded_img.resize((new_width, new_height))

    st.sidebar.text('Query Image', help="Edit the bounding box to change the ROI (Region of Interest).")
    with st.sidebar.empty():
        cropped_img = st_cropper(uploaded_img, box_color="#4fc4f9", realtime_update=True, aspect_ratio=(16,9))
 
    # Manipulate cropped image at will (optional, not sure if it is necessary)
    # adding it may cause slider or other functions not seen in sidebar
    # st.sidebar.write("Preview of selected region")
    # st.sidebar.image(cropped_img, use_column_width=True)

    show_distance = st.sidebar.toggle("Show Distance")

    # top k value slider
    value = st.sidebar.slider("Select top k results shown", 10, 100, 20, step = 1)

    
    @st.cache_resource
    def get_image_embedding(image_path):
        logger.info("Extracting image features")
        return extractor(image_path)
        
    image_embedding = get_image_embedding(cropped_img)

    results = client.search(
        "image_embeddings",
        data=[extractor(cropped_img)],
        limit = value,
        output_fields=["filename"],
        search_params={"metric_type": "COSINE"},
    )
    search_results = results[0]

    for i, info in enumerate(search_results):
        img_info = info["entity"]
        imgName = img_info["filename"]
        score = info["distance"]
        img = Image.open(imgName)
        cols[i % 5].image(img, use_column_width=True)
        if show_distance:
            cols[i % 5].write(f"Score: {score:.3f}")