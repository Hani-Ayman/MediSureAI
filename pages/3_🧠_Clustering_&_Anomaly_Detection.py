import os
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Configure Streamlit page
st.set_page_config(page_title="Clustering & Anomaly Detection", layout="wide")
st.title("📊 Image Clustering & Anomaly Detection (DBSCAN & KMeans)")

# 🔹 Root folder input
st.sidebar.subheader("⚙️ Settings")
root_folder = st.sidebar.text_input("Root Folder Path", 
    value=r"Enter path to images")

if not os.path.exists(root_folder):
    st.error("❌ Root folder not found. Please check the path.")
    st.stop()

# 🔹 Detect subfolders as categories
categories = sorted([f for f in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, f))])
if not categories:
    st.error("❌ No image folders found.")
    st.stop()

selected_category = st.sidebar.selectbox("Select Category", categories)
selected_folder = os.path.join(root_folder, selected_category)
st.success(f"✅ Using folder: {selected_folder}")

# 🔹 Load and preprocess images
@st.cache_data
def load_images(folder_path, target_size=(128, 128)):
    paths, data = [], []
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, file in enumerate(files):
        try:
            img_path = os.path.join(folder_path, file)
            img = load_img(img_path, target_size=target_size)
            paths.append(img_path)
            data.append(img_to_array(img))

            progress = (i + 1) / len(files)
            progress_bar.progress(progress)
            status_text.text(f"Loading images... {int(progress * 100)}%")
        except Exception as e:
            st.warning(f"⚠️ Failed to load {file}: {e}")

    progress_bar.empty()
    status_text.empty()
    return np.array(data), paths

images, image_paths = load_images(selected_folder)
if len(images) == 0:
    st.warning("⚠️ No valid images found.")
    st.stop()

# 🔹 Feature extraction using MobileNetV2
@st.cache_data
def extract_features(_images):
    model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(128, 128, 3))
    _images = preprocess_input(_images.astype('float32'))
    features = model.predict(_images, verbose=0)
    features_scaled = StandardScaler().fit_transform(features)
    return PCA(n_components=2).fit_transform(features_scaled)

features_pca = extract_features(images)

# 🔹 Sidebar clustering controls
st.sidebar.subheader("📌 Clustering Controls")
eps_val = st.sidebar.slider("DBSCAN eps", 0.1, 10.0, 3.0, step=0.1)
min_samples = st.sidebar.slider("DBSCAN min_samples", 1, 20, 5)
k = st.sidebar.slider("KMeans clusters", 2, 10, 3)
n_std = st.sidebar.slider("Anomaly threshold (σ)", 1.0, 3.0, 2.0, step=0.5)

@st.cache_data
def perform_clustering(_features, eps, min_samples, n_clusters):
    db = DBSCAN(eps=eps, min_samples=min_samples)
    db_labels = db.fit_predict(_features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(_features)

    return db_labels, kmeans_labels, kmeans

db_labels, kmeans_labels, kmeans = perform_clustering(features_pca, eps_val, min_samples, k)

# 🔹 Anomaly Detection
def detect_anomalies_dbscan(labels):
    return np.where(labels == -1)[0].tolist()

def detect_anomalies_kmeans(labels, features, model, threshold):
    anomalies = []
    centroids = model.cluster_centers_
    for i, (label, point) in enumerate(zip(labels, features)):
        centroid = centroids[label]
        dist = np.linalg.norm(point - centroid)
        cluster_points = features[labels == label]
        cluster_distances = [np.linalg.norm(p - centroid) for p in cluster_points]
        cluster_threshold = np.mean(cluster_distances) + threshold * np.std(cluster_distances)
        if dist > cluster_threshold:
            anomalies.append(i)
    return anomalies

db_anomalies = detect_anomalies_dbscan(db_labels)
kmeans_anomalies = detect_anomalies_kmeans(kmeans_labels, features_pca, kmeans, n_std)
combined_anomalies = list(set(db_anomalies) & set(kmeans_anomalies))

# 🔹 Cluster plot
def plot_clusters(pca, labels, anomalies, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    normal_mask = ~np.isin(np.arange(len(labels)), anomalies)

    for lbl in set(labels[normal_mask]):
        if lbl == -1:
            continue
        xy = pca[labels == lbl]
        ax.scatter(xy[:, 0], xy[:, 1], label=f"Cluster {lbl}", alpha=0.7)

    if anomalies:
        anomaly_points = pca[anomalies]
        ax.scatter(anomaly_points[:, 0], anomaly_points[:, 1],
                   color='red', marker='X', s=100,
                   label='Anomalies', linewidths=1.5)

    ax.set_title(title)
    ax.legend()
    return fig

# 🔹 Image display
def display_image_grid(indices, paths, title, max_images=12):
    st.subheader(title)
    if not indices:
        st.info("No images found")
        return

    n_cols = 4
    n_rows = int(np.ceil(min(len(indices), max_images) / n_cols))
    for row in range(n_rows):
        cols = st.columns(n_cols)
        for col in range(n_cols):
            idx = row * n_cols + col
            if idx < len(indices) and idx < max_images:
                try:
                    img = Image.open(paths[indices[idx]]).convert("RGB")
                    cols[col].image(img, use_container_width=True,
                                    caption=f"Image {indices[idx]+1}")
                except Exception as e:
                    cols[col].error(f"Error loading image: {e}")

# 🔹 Tabs
tab1, tab2, tab3 = st.tabs(["Cluster Visualization", "Anomaly Detection", "Image Explorer"])

with tab1:
    st.header("📍 Cluster Visualization")
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_clusters(features_pca, db_labels, db_anomalies, "DBSCAN Clustering"))
    with col2:
        st.pyplot(plot_clusters(features_pca, kmeans_labels, kmeans_anomalies, "KMeans Clustering"))

with tab2:
    st.header("🚨 Anomaly Detection Results")
    col1, col2, col3 = st.columns(3)
    col1.metric("DBSCAN Anomalies", len(db_anomalies))
    col2.metric("KMeans Anomalies", len(kmeans_anomalies))
    col3.metric("High-Confidence Anomalies", len(combined_anomalies))

    anomaly_type = st.radio("Show anomalies by:", ["DBSCAN only", "KMeans only", "Both methods"], horizontal=True)
    if anomaly_type == "DBSCAN only":
        display_image_grid(db_anomalies, image_paths, "DBSCAN-Detected Anomalies")
    elif anomaly_type == "KMeans only":
        display_image_grid(kmeans_anomalies, image_paths, "KMeans-Detected Anomalies")
    else:
        display_image_grid(combined_anomalies, image_paths, "High-Confidence Anomalies")

with tab3:
    st.header("🔍 Image Explorer")
    cluster_method = st.radio("Clustering method:", ["DBSCAN", "KMeans"], horizontal=True)
    if cluster_method == "DBSCAN":
        clusters = sorted(set(db_labels))
        selected_cluster = st.selectbox("Select cluster", clusters)
        cluster_indices = np.where(db_labels == selected_cluster)[0].tolist()
    else:
        clusters = sorted(set(kmeans_labels))
        selected_cluster = st.selectbox("Select cluster", clusters)
        cluster_indices = np.where(kmeans_labels == selected_cluster)[0].tolist()

    display_image_grid(cluster_indices, image_paths, f"Images in Cluster {selected_cluster}")

# 🔹 Export report
st.sidebar.subheader("📁 Export")
if st.sidebar.button("📥 Save Anomaly Report"):
    report = f"Anomaly Detection Report\n\n"
    report += f"Category: {selected_category}\n"
    report += f"Total Images: {len(images)}\n\n"
    report += f"DBSCAN Parameters: eps={eps_val}, min_samples={min_samples}\n"
    report += f"DBSCAN Anomalies: {len(db_anomalies)}\n\n"
    report += f"KMeans Parameters: clusters={k}, threshold={n_std}σ\n"
    report += f"KMeans Anomalies: {len(kmeans_anomalies)}\n\n"
    report += f"High-Confidence Anomalies: {len(combined_anomalies)}\n\n"

    if combined_anomalies:
        report += "Anomaly Files:\n"
        for idx in combined_anomalies:
            report += f"- {os.path.basename(image_paths[idx])}\n"

    with open("anomaly_report.txt", "w") as f:
        f.write(report)

    st.sidebar.success("Report saved as anomaly_report.txt")
