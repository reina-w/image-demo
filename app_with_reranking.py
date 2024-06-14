import streamlit as st
import streamlit_cropper
from streamlit_cropper import st_cropper
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
import numpy as np

# Reranking function 
def k_reciprocal_neigh(initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i, :k1 + 1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
    fi = np.where(backward_k_neigh_index == i)[0]
    return forward_k_neigh_index[fi]

def re_ranking(q_g_dist, q_q_dist, g_g_dist, k1, k2=6, lambda_value=0.3):
    original_dist = np.concatenate(
        [np.concatenate([q_q_dist, q_g_dist], axis=1),
         np.concatenate([q_g_dist.T, g_g_dist], axis=1)],
        axis=0)
    original_dist = 2. - 2 * original_dist
    original_dist = np.power(original_dist, 2).astype(np.float32)
    original_dist = np.transpose(1. * original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float32)
    initial_rank = np.argpartition(original_dist, range(1, min(k1 + 1, original_dist.shape[1])))

    query_num = q_g_dist.shape[0]
    all_num = original_dist.shape[0]

    for i in range(all_num):
        k_reciprocal_index = k_reciprocal_neigh(initial_rank, i, k1)
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_k_reciprocal_index = k_reciprocal_neigh(initial_rank, candidate, int(np.around(k1 / 2)))
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2. / 3 * len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = 1. * weight / np.sum(weight)

    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float32)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(all_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float32)
    for i in range(query_num):
        temp_min = np.zeros(shape=[1, all_num], dtype=np.float32)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]], V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2. - temp_min)

    print('original', original_dist)
    print('jaccard', jaccard_dist)

    final_dist = jaccard_dist*(1-lambda_value) + original_dist*lambda_value
    # final_dist = np.power(jaccard_dist, 2) * (1 - lambda_value) + original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    print('final', final_dist)
    return final_dist

# Streamlit Cropper Customization
def _recommended_box2(img: Image, aspect_ratio: tuple = None) -> dict:
    width, height = img.size
    return {'left': int(width * 0.05), 'top': int(height * 0.05), 'width': int(width * 0.8), 'height': int(height * 0.9)}
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

st.title("Image Similarity Search :frame_with_picture:")

query_image = "temp.jpg"
cols = st.columns(5)

uploaded_file = st.sidebar.file_uploader("Choose an image...", type="jpeg")

if uploaded_file is not None:
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    uploaded_img = Image.open(uploaded_file)
    width, height = uploaded_img.size

    new_width = 370
    new_height = int((new_width / width) * height)
    uploaded_img = uploaded_img.resize((new_width, new_height))

    st.sidebar.text('Query Image', help="Edit the bounding box to change the ROI (Region of Interest).")
    with st.sidebar.empty():
        cropped_img = st_cropper(uploaded_img, box_color="#4fc4f9", realtime_update=True, aspect_ratio=(16,9))

    show_distance = st.sidebar.checkbox("Show Distance")

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
        limit=value,
        output_fields=["filename", "vector"],
        search_params={"metric_type": "COSINE"},
    )
    search_results = results[0]

    query_vector = extractor(cropped_img).reshape(1, -1)
    gallery_vectors = np.array([info["entity"]["vector"] for info in search_results])
    
    q_g_dist = np.linalg.norm(query_vector - gallery_vectors, axis=1).reshape(1, -1)
    q_q_dist = np.zeros((1, 1))  # since one query image, dist to itself is 0
    g_g_dist = np.linalg.norm(gallery_vectors[:, None] - gallery_vectors, axis=2)

    k1 = min(value, 20) 
    final_dist = re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=k1, k2=6, lambda_value=0.3)

    reranked_indices = np.argsort(final_dist[0])[:value]

    for i in reranked_indices:
        info = search_results[i]
        img_info = info["entity"]
        imgName = img_info["filename"]
        score = final_dist[0, i]
        img = Image.open(imgName)
        cols[i % 5].image(img, use_column_width=True)
        if show_distance:
            cols[i % 5].write(f"Score: {score:.3f}")
