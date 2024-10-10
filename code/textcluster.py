import json
import os
import random
import shutil
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torch import cosine_similarity
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from PIL import Image


#由于历史遗留问题，这个函数暂时留在这里
def process_MathVista_texts(output_dir="D:/datasets/MATHVISION/JSON"):
    ds = load_dataset("AI4Math/MathVista")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    texts = []
    image_paths = []

    # 遍历数据集中的每个数据条目
    for split in ['testmini', 'test']:
        for item in ds[split]:
            image_path = item['image']
            image_name = os.path.splitext(os.path.basename(image_path))[0]  # 提取图片文件名（无扩展名）
            json_filename = f"{image_name}.json"
            json_filepath = os.path.join(output_dir, json_filename)

            # 构建问答对的字典
            qa_data = {
                'question': item['question'],
                'answer': item['answer']
            }

            texts.append(item['question'] + " " + item['answer'])
            image_paths.append(image_path)

            # 将问答对写入对应的 JSON 文件中
            with open(json_filepath, 'w', encoding='utf-8') as json_file:
                json.dump(qa_data, json_file, ensure_ascii=False, indent=4)

    return texts, image_paths

#由于历史遗留问题，这个函数暂时留在这里
def process_Geo_texts(src_folder= "D:/datasets/GEO170K/images/train/geoqa_plus"):
    def extract_random_qa_pair(json_data):
        # Find all QA pairs
        qa_pairs = [
            {"question": entry['value'], "answer": json_data[i + 1]['value']}
            for i, entry in enumerate(json_data[:-1])
            if entry['from'] == 'human' and json_data[i + 1]['from'] == 'gpt'
        ]

        # Randomly select one QA pair
        random_pair = random.choice(qa_pairs)
        return random_pair

    def qa2qa(json_data):
        # 提取问题和答案
        question_data = json_data['question']
        answer_data = json_data['answer']

        # 提取问题部分
        question_start = question_data.find("Question:") + len("Question:")
        question_end = question_data.find("Choices:")
        question = question_data[question_start:question_end].strip()

        # 提取答案部分
        answer_start = answer_data.find("Since")
        answer_end = answer_data.find("Answer:")
        answer = answer_data[answer_start:answer_end].strip()

        answer_end = answer_data.find("Answer:")
        answer = answer_data[:answer_end].strip()
        answer = answer.replace("Figure 1", "this image")
        # 构建自然语言问答格式
        natural_language_qa = f'$$Q:"{question}"\n\n$$A:"{answer}"'

        return natural_language_qa
    annot_dir = "D:/datasets/GEO170K/images/train/JSON"
    texts = []
    image_paths = []

    # Iterate over all files in the image directory
    for image_file in os.listdir(src_folder):
        if image_file.endswith(".png"):
            image_name = os.path.splitext(image_file)[0]
            image_path = os.path.join(src_folder, image_file)

            # Construct the path to the corresponding JSON file
            json_folder = os.path.join(annot_dir, image_name)
            json_file = os.path.join(json_folder, "1.json")

            # Check if the JSON file exists
            if os.path.exists(json_file):
                with open(json_file, 'r', encoding='utf-8') as f:
                    text = json.load(f)
                qa_pair=extract_random_qa_pair(text)

                text=qa2qa(qa_pair)

                texts.append(text)

                image_paths.append(image_path)

    return texts, image_paths




def extract_text_features(texts, model_name='roberta-large'):
    model = SentenceTransformer(model_name)
    features = model.encode(texts, convert_to_tensor=True)
    return features


def perform_clustering(features, n_clusters=10, n_components=50):
    """
      使用PCA降维并执行K-means聚类。

      参数:
          features (Tensor): 文本特征张量。
          n_clusters (int): 聚类的簇数，默认为10。
          n_components (int): PCA降维后的特征维度数，默认为50。

      返回:
          clusters (dict): 每个类别对应的数据点索引字典。
          kmeans (KMeans): K-means模型对象。
      """
    # 使用PCA进行降维
    pca = PCA(n_components=n_components)
    # 将特征从 GPU 移到 CPU
    features = features.cpu().numpy()
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

    return clusters, kmeans


def save_selected_images(clusters, centroids, image_paths, output_base_folder, src_folder, n_groups=20, texts=None, features=None):
    """
    保存选定聚类中的图片，并为每对图片生成JSON文件。

    参数:
        clusters (dict): 每个聚类中包含的数据点索引字典。
        centroids (array): 每个聚类中心的坐标。
        image_paths (list): 原始图片的文件路径列表。
        output_base_folder (str): 输出文件夹路径。
        src_folder (str): 图片源文件夹路径。
        n_groups (int): 选择的聚类数量，默认为20。
        texts (list, optional): 对应的文本列表，用于保存相关文本数据。
        features (Tensor, optional): 特征张量，用于计算图片对的相似度。

    返回:
        无返回值。结果保存到指定的输出文件夹中。
    """
    # 筛选数量大于3的聚类
    valid_clusters = {k: v for k, v in clusters.items() if len(v) > 3}

    # 计算每个聚类中心到原点的距离，并按距离排序
    distances = {k: np.linalg.norm(centroids[k]) for k in valid_clusters.keys()}
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

        max_similarity = -float('inf')
        selected_pair = None

        for i in range(len(cluster_points)):
            for j in range(i + 1, len(cluster_points)):
                sim = cosine_similarity(features[cluster_points[i]].reshape(1,-1), features[cluster_points[j]].reshape(1,-1))
                print(sim)
                if sim > max_similarity:
                    max_similarity = sim
                    selected_pair = [cluster_points[i], cluster_points[j]]
        print(max_similarity)
        if selected_pair is None:
            continue

        cluster_folder = os.path.join(output_base_folder, str(next_folder_idx))
        os.makedirs(cluster_folder, exist_ok=True)

        for img_idx, point_idx in enumerate(selected_pair, start=1):
            src = os.path.join(src_folder, image_paths[point_idx])
            dst = os.path.join(cluster_folder, f'{img_idx}.jpg')
            shutil.copy(src, dst)

            # Save corresponding text as a .txt file
            if texts:
                idx=os.path.splitext(os.path.basename(image_paths[point_idx]))[0]
                src = os.path.join(f"D:/datasets/GEO170K/images/train/JSON/{idx}",'1.json')
                dst = os.path.join(cluster_folder,f'{img_idx}.json')
                shutil.copy(src, dst)
                # text_path = os.path.join(cluster_folder, f'{img_idx}.txt')
                # with open(text_path, 'w', encoding='utf-8') as text_file:
                #     text_file.write(str(texts[point_idx]))

        next_folder_idx += 1



def main(output_base_folder,src_folder, n_clusters=10, n_groups=5):
    """
        主函数，执行文本处理、特征提取、聚类及图片保存。

        参数:
            output_base_folder (str): 输出文件夹路径。
            src_folder (str): 图片源文件夹路径。
            n_clusters (int): 聚类的簇数，默认为10。
            n_groups (int): 选择的聚类数量，默认为5。

        返回:
            无返回值，结果保存到指定的输出文件夹中。
        """
    #texts, image_paths = process_MathVista_texts()
    texts, image_paths = process_Geo_texts(src_folder)
    # 提取文本特征
    features = extract_text_features(texts)

    # 执行聚类
    clusters, kmeans = perform_clustering(features, n_clusters, n_components=50)

    # 获取每个类别的聚类中心
    centroids = kmeans.cluster_centers_

    # 保存选定的图像
    save_selected_images(clusters, centroids, image_paths, output_base_folder, src_folder, n_groups, texts=texts,features=features)



if __name__ == "__main__":
    output_base_folder = "GEO"
    src_folder = 'D:/datasets/GEO170K/images/train/geoqa_plus'
    main(output_base_folder, src_folder, n_clusters=10, n_groups=5)