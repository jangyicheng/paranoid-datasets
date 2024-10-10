import re

import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.cluster import KMeans
import os
import shutil
from sklearn.decomposition import PCA
import random
from PIL import Image
from scipy.spatial.distance import cdist
from tqdm import tqdm


# 自定义数据集类，用于加载图像数据
class ImageDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if
                            img.endswith(('jpg', 'jpeg', 'png'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

def extract_features(image_folder, transform, device):
    dataset = ImageDataset(image_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    # 使用预训练的ResNet模型提取特征
    model = models.resnet50(pretrained=True)
    model = model.eval().to(device)

    features = []

    with torch.no_grad():
        for images in dataloader:
            images = images.to(device)
            outputs = model(images)
            features.append(outputs.cpu().numpy())

    features = np.concatenate(features, axis=0)
    return features, dataset.image_paths


def perform_clustering(features, n_clusters=10, n_components=50):
    # 使用PCA进行降维
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features)

    # 使用K-means进行聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(reduced_features)
    cluster_labels = kmeans.labels_

    # 初始化存储每个类别的数据点的列表
    clusters = {i: [] for i in range(n_clusters)}

    # 将每个数据点添加到对应的类别中
    for i, label in enumerate(cluster_labels):
        clusters[label].append(i)

    return clusters,kmeans


def save_selected_images(clusters, centroids, image_paths, output_base_folder, n_groups=20):
    # 筛选数量大于3的聚类
    valid_clusters = {k: v for k, v in clusters.items() if len(v) > 3}

    # 计算每个聚类中心到原点的距离，并按距离排序
    distances = {k: cdist([centroids[k]], [[0] * centroids.shape[1]])[0][0] for k in valid_clusters.keys()}
    sorted_clusters = sorted(distances.keys(), key=lambda x: distances[x])

    # 选择前n_groups个聚类
    selected_clusters = sorted_clusters[:n_groups]

    # 获取当前最大的文件夹编号
    if not os.path.exists(output_base_folder):
        os.makedirs(output_base_folder)
    existing_dirs = [int(d) for d in os.listdir(output_base_folder) if d.isdigit()]
    next_folder_idx = max(existing_dirs, default=0) + 1

    for cluster_label in tqdm(selected_clusters):
        cluster_points = valid_clusters[cluster_label]
        selected_points = random.sample(cluster_points, random.randint(3, 5))
        cluster_folder = os.path.join(output_base_folder, str(next_folder_idx))
        os.makedirs(cluster_folder, exist_ok=True)

        for img_idx, point_idx in enumerate(selected_points, start=1):
            src = image_paths[point_idx]
            dst = os.path.join(cluster_folder, f'{img_idx}.jpg')
            shutil.copy(src, dst)

        next_folder_idx += 1

def main(base_folder, output_base_folder,n_clusters=10,n_group=10):#对于指定目录下所有文件夹进行遍历，每一文件夹进行10个聚类，筛出前n_group个聚类
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(output_base_folder):
        os.makedirs(output_base_folder)
    for subdir in os.listdir(base_folder):
        if os.path.basename(subdir)!='JSON':
            image_folder = os.path.join(base_folder, subdir)
            if os.path.isdir(image_folder):
                print(f"Processing folder: {image_folder}")
                features, image_paths = extract_features(image_folder, transform, device)
                clusters ,kmeans = perform_clustering(features, n_clusters, n_components=50)
                # 获取每个类别的聚类中心
                centroids = kmeans.cluster_centers_
                save_selected_images(clusters, centroids,image_paths, output_base_folder)

if __name__ == "__main__":

    base_folder = "D:/datasets/GEO170K/images/train/"
    output_base_folder = "GEO170K_"
    main(base_folder, output_base_folder,n_clusters=10,n_group=5)









