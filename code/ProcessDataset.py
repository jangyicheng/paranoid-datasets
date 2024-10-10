import io
import json
import os
import random
import re
import shutil
from collections import defaultdict
from urllib.error import HTTPError, ContentTooShortError
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import requests
from PIL import Image
import pyarrow.parquet as pq
from PyPDF2 import PdfFileReader
from datasets import load_dataset
from matplotlib import pyplot as plt
from pdf2image import convert_from_path
from tqdm import tqdm


def Process_ScreenQA(parquet_files, images_dir, answers_dir):
    """
    这里只是预处理，后续需要聚类操作进行分组
    """
    def Preprocess_ScreenQA(parquet_files, images_dir='D:/datasets/SCREENQA/images', answers_dir='D:/datasets/SCREENQA/annotations'):
        # Create directories if they don't exist
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
        if not os.path.exists(answers_dir):
            os.makedirs(answers_dir)

        counter = 1

        for parquet_file in parquet_files:
            # Read the parquet file
            table = pq.read_table(parquet_file)
            df = table.to_pandas()

            for index, row in df.iterrows():
                # Extract image and answer
                image = row['image']
                answer = row['answer']

                # Save the image
                image_path = os.path.join(images_dir, f"{counter}.jpg")
                img_data = image['bytes']
                image = Image.open(io.BytesIO(img_data))
                image.save(image_path)

                # Save the answer
                answer_path = os.path.join(answers_dir, f"{counter}.txt")
                with open(answer_path, 'w', encoding='utf-8') as answer_file:
                    answer_file.write(answer)

                # Increment the counter
                counter += 1



def Process_Ai2d(source_dir='D:/datasets/AI2d/images/',
                 target_dir='AI2D',
                 json_file_path='D:/datasets/AI2d/categories.json',
                 annotation_dir='D:/datasets/AI2d/questions'):
    """
    处理AI2D图像数据集，将其根据类别分类并移动到新目录。

    参数：
    - source_dir (str): 图像文件的源目录路径。
    - target_dir (str): 分类后图像文件的目标目录路径。
    - json_file_path (str): 包含图像类别信息的JSON文件路径。
    - annotation_dir (str): 存放图像标注文件的目录路径。
    """

    # 创建目标目录，如果不存在则创建
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 读取JSON文件，获取每个图像的类别信息
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 收集所有类别的图片文件
    category_images = {}
    for image_file, category in data.items():
        if category not in category_images:
            category_images[category] = []
        category_images[category].append(image_file)

    folder_index = 1
    # 对每个类别的图片文件进行处理
    for category, files in category_images.items():
        random.shuffle(files)
        files = files[:50]  # 随机抽取前50个文件
        while len(files) > 5:
            num_files_to_select = random.randint(3, 5)
            selected_files = files[:num_files_to_select]
            files = files[num_files_to_select:]

            # 创建新的文件夹
            new_folder_path = os.path.join(target_dir, str(folder_index))
            os.makedirs(new_folder_path, exist_ok=True)

            for i, image_file in enumerate(selected_files, start=1):
                # 移动并重命名图片文件
                src_image_path = os.path.join(source_dir, image_file)
                tgt_image_path = os.path.join(new_folder_path, f"{i}{os.path.splitext(image_file)[1]}")
                shutil.move(src_image_path, tgt_image_path)

                # 移动并重命名JSON标注文件
                annotation_file = image_file + '.json'
                src_json_path = os.path.join(annotation_dir, annotation_file)
                tgt_json_path = os.path.join(new_folder_path, f"{i}.json")
                if os.path.exists(src_json_path):
                    shutil.move(src_json_path, tgt_json_path)

            folder_index += 1


def Process_Animals(src_directory=r"D:\datasets\ANIMALS\animals\animals", target_directory="ANIMALS", num_samples=4, min_images=3, max_images=5):
    # 创建目标目录
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    # 遍历src_directory下的所有子文件夹
    animal_folders = [os.path.join(src_directory, folder) for folder in os.listdir(src_directory) if os.path.isdir(os.path.join(src_directory, folder))]

    sample_count = 1

    for _ in range(num_samples):
        for animal_folder in animal_folders:
            # 获取当前动物文件夹下的所有图片
            images = [os.path.join(animal_folder, img) for img in os.listdir(animal_folder) if img.endswith(('.jpg', '.jpeg', '.png'))]

            # 从中随机选择3到5张图片
            num_images = random.randint(min_images, max_images)
            sampled_images = random.sample(images, num_images)

            # 为每个样本创建一个新的文件夹
            sample_directory = os.path.join(target_directory, str(sample_count))
            if not os.path.exists(sample_directory):
                os.makedirs(sample_directory)

            # 复制并重命名图片
            for idx, image in enumerate(sampled_images):
                dst_image_path = os.path.join(sample_directory, f"{idx + 1}.jpg")
                shutil.copy(image, dst_image_path)

            # 获取动物类别并写入1.txt
            animal_category = os.path.basename(animal_folder)
            with open(os.path.join(sample_directory, "1.txt"), "w") as f:
                f.write(animal_category)

            sample_count += 1


def Process_Nextqa(source_folder='D:/datasets/NEXTQA', dest_folder='NEXTQA', sample_size=200):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # Create a dictionary to group files by their prefix
    prefix_groups = defaultdict(list)

    # Traverse the source folder and group files by prefix
    for filename in os.listdir(source_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # You can add more file extensions if needed
            prefix = filename.split('_')[0]
            prefix_groups[prefix].append(filename)

    # Sample 200 prefix groups
    sampled_prefixes = random.sample(list(prefix_groups.keys()), min(sample_size, len(prefix_groups)))

    # Process each group and move files to the destination folder
    for i, prefix in enumerate(sampled_prefixes, start=1):
        group = prefix_groups[prefix]

        # Randomly keep 3 to 5 images in each group
        num_to_keep = random.randint(5, 6)
        sampled_files = random.sample(group, min(num_to_keep, len(group)))

        # Create a new folder for this group
        group_folder = os.path.join(dest_folder, str(i))
        os.makedirs(group_folder, exist_ok=True)

        # Move and rename files
        for j, filename in enumerate(sampled_files, start=1):
            src_path = os.path.join(source_folder, filename)
            # Extract the original file extension
            file_ext = os.path.splitext(filename)[1]
            dest_filename = f"{j}{file_ext}"
            dest_path = os.path.join(group_folder, dest_filename)
            shutil.copy(src_path, dest_path)


def Process_Web(parquet_files=["D:/datasets/MIND2WEB/web.parquet"], output_dir='MIND2WEB', num=100, max_height=800):
    """
    处理包含网页截图的Parquet文件，并将截图按指定的最大高度拆分成较小的图片。

    参数:
        parquet_files (list): 包含Parquet文件路径的列表，默认是 ["D:/datasets/MIND2WEB/web.parquet"]。
        output_dir (str): 存放处理后的截图文件的输出目录，默认是 'MIND2WEB'。
        num (int): 每个分组随机抽取的截图数量，默认是100。
        max_height (int): 拆分后的每张图片的最大高度，默认是800。

    返回:
        无返回值。处理后的截图被保存到指定的输出目录中。
    """

    def split_webpage_screenshot(image_path, output_folder, max_height=max_height):
        """
        将网页截图按最大高度拆分成多个较小的图片。

        参数:
            image_path (str): 原始截图的文件路径。
            output_folder (str): 存储拆分后的截图文件夹路径。
            max_height (int): 每张拆分图片的最大高度。

        返回:
            无返回值。拆分后的图片被保存到指定的输出文件夹中。
        """
        # 加载图像
        img = Image.open(image_path)
        width, height = img.size  # 获取图像的宽度和高度

        # 确保输出文件夹存在
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # 定义切割的起始点
        start_y = 0
        img_index = 1  # 图片编号

        # 循环切割图像，直到处理完成或达到最大图片数量（最多 4 张）
        while start_y < height and img_index < 5:
            # 确定当前切片的结束位置
            end_y = min(start_y + max_height, height)

            # 裁剪图像
            cropped_img = img.crop((0, start_y, width, end_y))

            # 保存裁剪后的图像
            cropped_img.save(os.path.join(output_folder, f"{img_index}.png"))

            # 更新起始位置，用于下一次切割
            start_y = end_y
            img_index += 1

    # 如果输出目录不存在，则创建它
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    folder_idx = 1  # 文件夹编号

    # 遍历所有Parquet文件
    for parquet_file in parquet_files:
        # 读取Parquet文件
        df = pd.read_parquet(parquet_file)

        # 按 website 和 subdomain 分组
        grouped = df.groupby(['website', 'subdomain'])

        # 遍历每个分组 (website, subdomain)
        for (website, subdomain), group in grouped:
            # 如果分组中的行数少于num，则调整num为实际行数
            n = min(num, len(group))

            # 随机抽取 num 张截图
            sampled_rows = group.sample(n=n)

            # 遍历抽取的行
            for i, row in sampled_rows.iterrows():
                screenshot_data = row['screenshot']['bytes']  # 获取截图数据

                # 将截图数据转换为图像
                screenshot_image = Image.open(io.BytesIO(screenshot_data))

                # 创建文件夹
                folder_path = os.path.join(output_dir, f"{folder_idx}")
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                # 保存截图
                image_path = os.path.join(folder_path, 'screenshot.png')
                screenshot_image.save(image_path, 'PNG')

                # 将截图拆分成小图
                split_webpage_screenshot(image_path, folder_path, max_height)

                # 拆分后删除原始截图
                os.remove(image_path)

                folder_idx += 1




def Process_MagicBrush(parquet_file_path="D:/datasets/MAGICBRUSH/magic_brush.parquet", output_dir='MAGICBRUSH'):
    # 读取Parquet文件
    df = pd.read_parquet(parquet_file_path)

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历每一行
    for index, row in df.iterrows():
        # 创建文件夹
        folder_path = os.path.join(output_dir, str(index + 1))
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 保存source_img
        source_img_bytes = row['source_img']['bytes']
        source_img_path = os.path.join(folder_path, '1.png')
        with open(source_img_path, 'wb') as f:
            f.write(source_img_bytes)

        # 保存target_img
        target_img_bytes = row['target_img']['bytes']
        target_img_path = os.path.join(folder_path, '2.png')
        with open(target_img_path, 'wb') as f:
            f.write(target_img_bytes)

        # 保存instruction
        instruction = row['instruction']
        instruction_path = os.path.join(folder_path, '2.txt')
        with open(instruction_path, 'w') as f:
            f.write(instruction)

def Process_Lec(data_file='D:/datasets/LectureBank/alldata.tsv', output_dir='LECTUREBANK', num_files=30):
    def download_pdf(url, path):
        response = requests.get(url)
        with open(path, 'wb') as file:
            file.write(response.content)

    # 读取TSV文件
    df = pd.read_csv(data_file, sep='\t')

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    valid_pdfs = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        url = row['URL']
        pdf_path = os.path.join(output_dir, url.split('/')[-1])

        if pdf_path.lower().endswith('.pdf'):
            try:
                # 下载PDF文件
                if not os.path.isfile(pdf_path):
                    download_pdf(url, pdf_path)

                # 读取PDF文件
                with open(pdf_path, 'rb') as f:
                    pdf = PdfFileReader(f)
                    num_pages = pdf.getNumPages()

                    if num_pages > 3:
                        valid_pdfs.append((pdf_path, num_pages))

                    # 如果已经收集到num_files个文件，停止
                    if len(valid_pdfs) >= num_files:
                        break
            except Exception as e:
                print(f"Error processing {url}: {e}")

    # 处理符合条件的PDF文件
    for folder_idx, (pdf_path, num_pages) in tqdm(enumerate(valid_pdfs), total=len(valid_pdfs)):
        # 创建文件夹
        folder_path = os.path.join(output_dir, str(folder_idx + 1))
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 检查文件名格式
        filename = os.path.basename(pdf_path)
        if filename.split('_')[0].isdigit():#这一类pdf文件页面关系形如四宫格
            # 分割图像为2x2的四张图片
            selected_page = random.randint(1, num_pages - 1)
            images = convert_from_path(pdf_path, first_page=selected_page + 1, last_page=selected_page + 1)
            image = images[0]

            # 获取图像尺寸
            width, height = image.size

            # 分割图像为2x2的四张图片
            coords = [
                (0, 0, width // 2, height // 2),
                (width // 2, 0, width, height // 2),
                (0, height // 2, width // 2, height),
                (width // 2, height // 2, width, height),
            ]
            for i, (left, upper, right, lower) in enumerate(coords, start=1):
                cropped_image = image.crop((left, upper, right, lower))
                image_name = os.path.join(folder_path, f"{i}.png")
                cropped_image.save(image_name, 'PNG')
        else:
            # 任意采样3-5页连续的PDF页面作为一组图片
            start_page = random.randint(1, num_pages - 5)
            end_page = start_page + random.randint(3, 5)
            images = convert_from_path(pdf_path, first_page=start_page, last_page=min(end_page, num_pages))

            for i, image in enumerate(images, start=1):
                image_name = os.path.join(folder_path, f"{i}.png")
                image.save(image_name, 'PNG')

def Process_Office(src_directories=None,target_directory="OFFICEHOME",num_samples_per_class=4,num_subdirectories=4):
    #解析office home数据集
    if src_directories is None:
        src_directories = [
            r"D:\datasets\OFFICEHOME\OfficeHomeDataset_10072016\Art",
            r"D:\datasets\OFFICEHOME\OfficeHomeDataset_10072016\Clipart",
            r"D:\datasets\OFFICEHOME\OfficeHomeDataset_10072016\Product",
            r"D:\datasets\OFFICEHOME\OfficeHomeDataset_10072016\Real World"
        ]

    # 创建目标目录
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    # 获取所有子类别文件夹的名称（假设四个主文件夹下的子类别文件夹名称一致）
    subdirectories = os.listdir(src_directories[0])

    sample_count = 1

    for subdirectory in subdirectories:
        subdir_paths = [os.path.join(src_dir, subdirectory) for src_dir in src_directories]

        for _ in range(num_samples_per_class):
            sampled_images = []

            for subdir_path in subdir_paths:
                if os.path.exists(subdir_path):
                    images = [os.path.join(subdir_path, img) for img in os.listdir(subdir_path) if
                              img.endswith(('.jpg', '.jpeg', '.png'))]
                    sampled_image = random.choice(images)
                    sampled_images.append(sampled_image)

            # 创建一个新的文件夹存放当前组的图片
            sample_directory = os.path.join(target_directory, str(sample_count))
            if not os.path.exists(sample_directory):
                os.makedirs(sample_directory)

            # 复制并重命名图片
            for idx, image in enumerate(sampled_images):
                dst_image_path = os.path.join(sample_directory, f"{idx + 1}.jpg")
                shutil.copy(image, dst_image_path)

            sample_count += 1


def Process_Resisc(output_folder= 'RESISC'):#解析RESISC数据集
    def read_parquet_files(parquet_paths):  # 读取parquet文件为pandas对象
        dfs = [pq.read_table(path).to_pandas() for path in parquet_paths]
        return pd.concat(dfs, ignore_index=True)
    parquet_paths = ['D:/360Downloads/train-00000-of-00001.parquet',
                     'D:/360Downloads/validation-00000-of-00001.parquet']  # 替换为实际Parquet文件路径
    df = read_parquet_files(parquet_paths)
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取类别到名称的映射
    label_names = {
        0: 'airplane', 1: 'airport', 2: 'baseball_diamond', 3: 'basketball_court', 4: 'beach',
        5: 'bridge', 6: 'chaparral', 7: 'church', 8: 'circular_farmland', 9: 'cloud',
        10: 'commercial_area', 11: 'dense_residential', 12: 'desert', 13: 'forest', 14: 'freeway',
        15: 'golf_course', 16: 'ground_track_field', 17: 'harbor', 18: 'industrial_area', 19: 'intersection',
        20: 'island', 21: 'lake', 22: 'meadow', 23: 'medium_residential', 24: 'mobile_home_park',
        25: 'mountain', 26: 'overpass', 27: 'palace', 28: 'parking_lot', 29: 'railway',
        30: 'railway_station', 31: 'rectangular_farmland', 32: 'river', 33: 'roundabout', 34: 'runway',
        35: 'sea_ice', 36: 'ship', 37: 'snowberg', 38: 'sparse_residential', 39: 'stadium',
        40: 'storage_tank', 41: 'tennis_court', 42: 'terrace', 43: 'thermal_power_station', 44: 'wetland'
    }

    # 按类别分类图像
    category_to_images = {label: [] for label in label_names.keys()}
    for idx, row in df.iterrows():
        label = row['label']
        image = row['image']
        category_to_images[label].append(image)

    # 对每个类别创建四个文件夹并进行随机采样
    for cat_id, img_files in category_to_images.items():
        cat_name = label_names[cat_id]
        random.shuffle(img_files)
        for folder_idx in range(1, 5):
            if len(img_files) < 5:  # 确保每类别至少有12张图片
                continue
            cat_folder = os.path.join(output_folder, f"{cat_id}_{folder_idx}")
            if not os.path.exists(cat_folder):
                os.makedirs(cat_folder)

            # 每个文件夹中采样3到5张图像
            sample_size = random.randint(3, 5)
            sampled_images = img_files[:sample_size]
            img_files = img_files[sample_size:]  # 更新图像列表，避免重复

            for idx, img_data in enumerate(sampled_images):
                img_data = img_data['bytes']
                img = Image.open(io.BytesIO(img_data))
                save_path = os.path.join(cat_folder, f"{idx + 1}.jpg")
                img.save(save_path)
                print(f"Saved {save_path}")

def Process_Comics(dataset="COMICS_Dialogue", target_directory='COMICS'):#对含有full.json的数据集进行解析
    # 创建目标目录
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    json_file_path=os.path.join("D:/datasets",dataset,"full/full.json" )
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    samples = data['data']

    # 随机采样200组数据
    samples = random.sample(samples, 100)
    for sample in samples:
        sample_id = sample['sample_id']
        images_path = sample['task_instance']['images_path']
        response = sample['response']
        # 为每个样本创建一个新的文件夹
        sample_directory = os.path.join(target_directory, str(sample_id + 1))
        if not os.path.exists(sample_directory):
            os.makedirs(sample_directory)

        for idx, image in enumerate(images_path[:-1]):
            src_image_path = os.path.join("D:/datasets",dataset,"full/images",image) # 假设图片路径是相对路径
            dst_image_path = os.path.join(sample_directory, f"{idx + 1}.jpg")

            # 复制图片到目标目录
            shutil.copy(src_image_path, dst_image_path)
        # if dataset == "COMICS_Dialogue":
        #     num_images = len(images_path)
        #     response_file_path = os.path.join(sample_directory, f"{num_images}.txt")
        #     with open(response_file_path, 'w', encoding='utf-8') as response_file:
        #         response_file.write("The "+response)


def Process_Alfred(dataset="ALFRED", target_directory='ALFRED'):
    # 创建目标目录
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    json_file_path = os.path.join("D:/datasets", dataset, "full/full.json")
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    samples = data['data']
    folder_counter = 1
    successful_samples = 0

    for i in range(len(samples) - 1):
        if successful_samples >= 100:
            break

        current_sample = samples[i]
        next_sample = samples[i + 1]
        next_images_path = next_sample['task_instance']['images_path']
        images_path = current_sample['task_instance']['images_path']
        # 仅当下一元素的 images_path 列表中仅含一个对象时处理
        if len(next_images_path) == 1:
            # 为每个样本创建一个新的文件夹
            sample_directory = os.path.join(target_directory, str(folder_counter))
            if not os.path.exists(sample_directory):
                os.makedirs(sample_directory)

            for idx, image in enumerate(images_path):
                src_image_path = os.path.join("D:/datasets", dataset, "full/images", image)  # 假设图片路径是相对路径
                dst_image_path = os.path.join(sample_directory, f"{idx + 1}{os.path.splitext(image)[1]}")

                # 复制图片到目标目录
                shutil.copy(src_image_path, dst_image_path)

            # 提取 context 和 response 并拼接
            context = current_sample['task_instance']['context']
            response = current_sample['response']
            text_content = f"{context} {response}"

            # 将 text_content 按照 '{image#' 进行切分
            steps = text_content.split('{image#')

            # 将 "Your Main Goal"、"Step Details" 和第一个步骤存入 1.txt
            main_goal_and_first_step = steps[0].strip() + ' {image#' + steps[1].strip()
            text_file_path = os.path.join(sample_directory, "1.txt")
            with open(text_file_path, 'w', encoding='utf-8') as text_file:
                text_file.write(main_goal_and_first_step)

            # 遍历后续的步骤并写入对应的文件
            for j, step in enumerate(steps[2:], start=2):
                step_text = '{image#' + step.strip()

                # 构建文件名
                text_file_path = os.path.join(sample_directory, f"{j}.txt")

                # 将步骤写入文件
                with open(text_file_path, 'w', encoding='utf-8') as text_file:
                    text_file.write(step_text)

            folder_counter += 1
            successful_samples += 1


def Process_Vasr(input_dir="D:/datasets/VASR/vasr_images", output_dir='VASR'):#解析VASR数据集
    # 创建输出目录
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 读取所有图片文件
    images = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]

    # 分类图片
    category_images = {}
    for img in images:
        category = img.split('_')[0]
        if category not in category_images:
            category_images[category] = []
        category_images[category].append(img)

    # 选择每个类别的3-4张图片，直到不足4张
    folder_idx = 1
    for category, imgs in category_images.items():
        while len(imgs) >= 4:
            selected_imgs = random.sample(imgs, random.randint(3, 4))  # 随机选择3-4张图片
            category_dir = os.path.join(output_dir, str(folder_idx))
            os.makedirs(category_dir, exist_ok=True)

            for img_idx, img_name in enumerate(selected_imgs, start=1):
                src_path = os.path.join(input_dir, img_name)
                dst_path = os.path.join(category_dir, f"{img_idx}.jpg")
                shutil.copy(src_path, dst_path)

                # 将类别名写入对应的文本文件
                text_path = os.path.join(category_dir, f"{img_idx}.txt")
                with open(text_path, 'w', encoding='utf-8') as text_file:
                    text_file.write(category)

            # 从列表中移除已经选择的图片
            for img in selected_imgs:
                imgs.remove(img)

            folder_idx += 1

    # 打印总图片数和总类别数
    total_images = sum(len(imgs) for imgs in category_images.values())
    total_categories = len(category_images)
    print(f"Total images: {total_images}")
    print(f"Total categories: {total_categories}")

def Process_Coco(images_folder='D:/datasets/COCO/train2017/', output_folder='COCO', num=4):
    # 注意：存在下载失败的bug待修复
    def download_image(url, save_path, retries=3):
        attempt = 0
        while attempt < retries:
            try:
                urlretrieve(url, save_path)
                return True
            except (HTTPError, ContentTooShortError) as e:
                print(f"Error: {e} while downloading {url}")
                attempt += 1
                if attempt == retries:
                    print(f"Failed to download {url} after {retries} attempts.")
                    return False
            except Exception as e:
                print(f"Error: {e} while downloading {url}")
                attempt += 1
                if attempt == retries:
                    print(f"Failed to download {url} after {retries} attempts.")
                    return False
        return False

    # 示例使用
    annotation_file = 'D:/datasets/COCO/annotation/annotations/instances_train2017.json'
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)
    categories = coco_data['categories']
    annotations = coco_data['annotations']
    images = coco_data['images']

    # 创建类别ID到名称的映射
    category_id_to_name = {cat['id']: cat['name'] for cat in categories}

    # 创建图片ID到文件名的映射
    image_id_to_file_name = {img['id']: img['file_name'] for img in images}

    # 创建图片ID到URL的映射
    image_id_to_url = {img['id']: img['coco_url'] for img in images}

    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 按类别分类的标注
    category_to_images = {cat['id']: [] for cat in categories}
    for ann in annotations:
        cat_id = ann['category_id']
        img_id = ann['image_id']
        if img_id in image_id_to_file_name:
            category_to_images[cat_id].append(image_id_to_file_name[img_id])

    folder_counter = 1

    # 对每个类别创建文件夹并进行随机采样
    for cat_id, img_files in category_to_images.items():
        if len(img_files) < 15:
            continue  # 跳过图像数量少于15的类别
        print(category_id_to_name[cat_id])
        cat_name = category_id_to_name[cat_id]
        random.shuffle(img_files)

        for folder_idx in range(1, num + 1):
            cat_folder = os.path.join(output_folder, f"{folder_counter}")
            folder_counter += 1
            if not os.path.exists(cat_folder):
                os.makedirs(cat_folder)

            # 每个文件夹中采样3到5张图像
            sample_size = random.randint(3, 5)
            sampled_images = img_files[:sample_size]
            img_files = img_files[sample_size:]
            for idx, img_file in enumerate(sampled_images):
                img_path = os.path.join(images_folder, img_file)
                save_path = os.path.join(cat_folder, f"{idx + 1}.jpg")
                # 如果图片在本地，直接复制
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    img.save(save_path)
                else:
                    # 如果图片不在本地，从URL下载
                    img_id = int(os.path.splitext(img_file)[0])
                    img_url = image_id_to_url[img_id]
                    if not download_image(img_url, save_path):
                        continue
                print(f"Saved {save_path}")

            # 创建并写入类别描述文件
            description_file_path = os.path.join(cat_folder, "1.txt")
            with open(description_file_path, 'w') as desc_file:
                desc_file.write(f"Each of the images in this set contains objects belonging to the category {cat_name}.")
            print(f"Saved {description_file_path}")


def Process_Doc(dataset="DOCVQA", target_directory='DOCVQA'):#对含有full.json的数据集进行解析
    # 创建目标目录
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    json_file_path=os.path.join("D:/datasets",dataset,"full/full.json" )
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    samples = data['data']

    # 随机采样200组数据
    samples = random.sample(samples, 100)
    for sample in samples:
        sample_id = sample['sample_id']
        images_path = sample['task_instance']['images_path']
        response = sample['response']
        # 为每个样本创建一个新的文件夹
        sample_directory = os.path.join(target_directory, str(sample_id + 1))
        if not os.path.exists(sample_directory):
            os.makedirs(sample_directory)

        for idx, image in enumerate(images_path):
            src_image_path = os.path.join("D:/datasets",dataset,"full/images",image) # 假设图片路径是相对路径
            dst_image_path = os.path.join(sample_directory, f"{idx + 1}.jpg")

            # 复制图片到目标目录
            shutil.copy(src_image_path, dst_image_path)


def Process_Ocr(dataset="OCR-VQA", target_directory='OCR'):#对含有full.json的数据集进行解析
    def rename_folders_in_sequence(folder_path):  # 将所有文件夹重新命名为1、2、3......
        # 获取指定文件夹中的所有子文件夹
        folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
        # 随机打乱文件夹列表
        random.shuffle(folders)
        # 对文件夹按顺序进行重新命名
        for i, folder_name in enumerate(folders, start=1):
            old_path = os.path.join(folder_path, folder_name)
            new_name = str(i)
            new_path = os.path.join(folder_path, new_name)

            # 检查新文件夹名是否已存在，如果存在，则添加后缀序号
            suffix = 1
            while os.path.exists(new_path):
                new_path = os.path.join(folder_path, new_name + f"_{suffix}")
                suffix += 1

            # 重命名文件夹
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} -> {new_path}")

        # 删除所有文件夹名中的后缀
        for folder_name in os.listdir(folder_path):
            old_path = os.path.join(folder_path, folder_name)
            new_name = re.sub(r"_\d+$", "", folder_name)  # 使用正则表达式删除后缀部分
            new_path = os.path.join(folder_path, new_name)

            if old_path != new_path:
                os.rename(old_path, new_path)
                print(f"Removed suffix: {old_path} -> {new_path}")
    # 创建目标目录
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    json_file_path=os.path.join("D:/datasets",dataset,"full/full.json" )
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    samples = data['data']

    # 随机采样200组数据
    samples = random.sample(samples, 100)
    for sample in samples:
        sample_id = sample['sample_id']
        images_path = sample['task_instance']['images_path']
        response = sample['response']
        # 为每个样本创建一个新的文件夹
        sample_directory = os.path.join(target_directory, str(sample_id + 1))
        if not os.path.exists(sample_directory):
            os.makedirs(sample_directory)

        for idx, image in enumerate(images_path):
            src_image_path = os.path.join("D:/datasets",dataset,"full/images",image) # 假设图片路径是相对路径
            dst_image_path = os.path.join(sample_directory, f"{idx + 1}.jpg")

            # 复制图片到目标目录
            shutil.copy(src_image_path, dst_image_path)
    rename_folders_in_sequence('OCR')


def Process_Wikiart(parquet_file_path="D:\\datasets\\WIKIART\\wikiart.parquet", output_dir='Wikiart'):
    # 读取Parquet文件
    df = pd.read_parquet(parquet_file_path)
    json_path="D:\\datasets\\WIKIART\\info.json"
    with open(json_path, 'r', encoding='utf-8') as file:
        info = json.load(file)["huggan--wikiart"]['features']

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 按artist和genre分组
    grouped = df.groupby(['artist', 'genre'])

    folder_index = 1
    for (artist, genre), group in grouped:
        # 取每组的前3张图片
        selected_images = group.head(3)

        # 创建文件夹
        folder_path = os.path.join(output_dir, str(folder_index))
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 保存图片
        for idx, row in enumerate(selected_images.itertuples(), start=1):
            image_bytes = row.image['bytes']
            image_path = os.path.join(folder_path, f'{idx}.jpg')
            with open(image_path, 'wb') as f:
                f.write(image_bytes)
            text_file_path = os.path.join(folder_path, f"{idx}.txt")
            with open(text_file_path, 'w', encoding='utf-8') as text_file:
                text_file.write(f'the artist is {info["artist"]["names"][artist]}, the genre is {info["genre"]["names"][genre]}')

        folder_index += 1


def Process_Med():
    # 加载数据集
    ds = load_dataset("FreedomIntelligence/PubMedVision", "PubMedVision_Alignment_VQA")

    # 创建输出基文件夹
    output_base_folder = 'PubMed'
    if not os.path.exists(output_base_folder):
        os.makedirs(output_base_folder)

    # 按 body_part 分组
    body_part_groups = ds['train'].to_pandas().groupby('body_part')

    for body_part, group in tqdm(body_part_groups):
        # 对每个 body_part 分组中的图片文件名按数字字符串排序
        group['sorted_images'] = group['image'].apply(
            lambda images: sorted(images, key=lambda x: os.path.basename(x))
        )
        # 只获取前50张图片
        selected_images = group.head(5000)

        folder_counter = 1
        image_counter = 1
        current_group_images = []

        for _, row in selected_images.iterrows():
            # 对于每一行，提取图片路径
            if len(row['image']) >= 2:
                continue
            for image_path in row['image']:

                image_filename = os.path.basename(image_path)
                image_extension = os.path.splitext(image_filename)[1]

                # 构建源图片路径
                src_image_path = os.path.join('D:/datasets/MED/images/', image_filename)

                # 检查源图片是否存在
                if not os.path.exists(src_image_path):
                    print(f"Image not found: {src_image_path}. Skipping...")
                    continue

                # 记录存在的图片路径和目标路径
                dst_image_path = os.path.join('PubMed', str(folder_counter), f"{image_counter}{image_extension}")
                current_group_images.append((src_image_path, dst_image_path, row['conversations']))

                image_counter += 1

                # 如果当前组有4张图片，则保存并重置计数器
                if len(current_group_images) >= 4:
                    folder_path = os.path.join(output_base_folder, str(folder_counter))
                    os.makedirs(folder_path, exist_ok=True)

                    for idx, (src, dst, conversations) in enumerate(current_group_images):
                        shutil.copy(src, dst)

                        # 保存对话文本
                        conversation_text = "\n".join([f"{conv['from']}: {conv['value']}" for conv in conversations])
                        text_file_path = os.path.join(folder_path, f"{idx + 1}.txt")
                        with open(text_file_path, 'w', encoding='utf-8') as text_file:
                            text_file.write(conversation_text)

                    current_group_images = []
                    folder_counter += 1
                    image_counter = 1

                    # 如果达到10组，停止处理该类别
                    if folder_counter > 10:
                        break
            # 检查是否已经达到10组
            if folder_counter > 10:
                break


def Process_Food101(parquet_files=["D:/datasets/FOOD101/food101_0.parquet", "D:/datasets/FOOD101/food101_1.parquet"], output_dir='FOOD'):
    folder_index = 1

    # 读取label.json文件
    with open('D:/datasets/FOOD101/label.json', 'r', encoding='utf-8') as f:
        label_mapping = json.load(f)['labels']

    for parquet_file in parquet_files:
        # 读取parquet文件
        df = pd.read_parquet(parquet_file)

        # 按label分组
        grouped = df.groupby('label')

        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for label, group in grouped:
            # 随机选择3-5张图片
            sample_images = group.sample(n=random.randint(3, 5))

            # 创建新的文件夹
            folder_path = os.path.join(output_dir, str(folder_index))
            os.makedirs(folder_path, exist_ok=True)

            # 保存图片
            for i, row in enumerate(sample_images.itertuples(), start=1):
                image_data = row.image['bytes']
                image = Image.open(io.BytesIO(image_data))
                image_path = os.path.join(folder_path, f"{i}.png")
                image.save(image_path)

            # 获取对应的类别名称
            category_name = label_mapping[str(label)]

            # 创建并写入文本文件
            text_file_path = os.path.join(folder_path, "1.txt")
            with open(text_file_path, 'w', encoding='utf-8') as text_file:
                text_file.write(f"The food in these pictures all belongs to the {category_name} category.")

            folder_index += 1

def Process_ScienceQA(parquet_file="D:/datasets/SCIENCEQA/scienceqa.parquet", root_dir='SCIENCEQA', min_images=3, max_images=5):
    # 解析ScienceQA文件
    # 读取parquet文件
    df = pd.read_parquet(parquet_file)

    # 创建根目录
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    # 获取所有的category
    categories = df['category'].unique()
    folder_index = 1

    # 按category处理图片
    for category in categories:
        category_df = df[df['category'] == category]
        images = category_df['image']
        print(category)
        count = 0

        while count != 5 and len(images) >= min_images:
            # 确定图片数量
            num_images = min(max_images, len(images))

            # 顺序选择图片
            selected_images = images.iloc[:num_images]

            # 检查是否有足够的有效图片
            valid_images = []
            for img in selected_images:
                if img and 'bytes' in img:
                    valid_images.append(img)
                if len(valid_images) >= min_images:
                    break

            if len(valid_images) < min_images:
                # 如果有效图片不足，继续读取更多图片
                images = images.iloc[num_images:]
                continue

            # 创建新的文件夹
            folder_path = os.path.join(root_dir, str(folder_index))
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            print(len(valid_images))

            # 保存图片
            for i, img in enumerate(valid_images):
                img_data = img['bytes']
                image = Image.open(io.BytesIO(img_data))
                image_file_path = os.path.join(folder_path, f"{i + 1}.jpg")
                image.save(image_file_path)

            # 移除已处理的图片
            images = images.iloc[num_images:]
            folder_index += 1
            print(f"save {category} to {folder_index}")

            count += 1

def Process_Natural(parquet_file="D:/datasets/Natural/naturalbench.parquet", output_dir='Natural'):
    # Read the Parquet file
    df = pd.read_parquet(parquet_file)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for index, row in df.iterrows():
        # Create a new folder for each row
        folder_path = os.path.join(output_dir, str(index + 1))
        os.makedirs(folder_path, exist_ok=True)

        # Save Image_0
        image_0 = Image.open(io.BytesIO(row['Image_0']['bytes']))
        image_0_path = os.path.join(folder_path, '1.jpg')  # Change extension if needed
        image_0.save(image_0_path)

        # Save Image_1
        image_1 = Image.open(io.BytesIO(row['Image_1']['bytes']))
        image_1_path = os.path.join(folder_path, '2.jpg')  # Change extension if needed
        image_1.save(image_1_path)

        # Save questions and answers for Image_0
        with open(os.path.join(folder_path, '1.txt'), 'w', encoding='utf-8') as f:
            f.write(f"Question: {row['Question_0']}\n")
            f.write(f"Answer: {row['Image_0_Question_0']}\n")

        # Save questions and answers for Image_1
        with open(os.path.join(folder_path, '2.txt'), 'w', encoding='utf-8') as f:
            f.write(f"Question: {row['Question_0']}\n")
            f.write(f"Answer: {row['Image_1_Question_0']}\n")


# Example usage

def Process(type=''):
    # Define a dictionary mapping types to their respective processing functions
    process_functions = {
        'scr': lambda: Process_ScreenQA(["D:/360Downloads/screenqa_large.parquet"], 'D:/datasets/SCREENQA/images', 'D:/datasets/SCREENQA/annotations'),
        'ai2d': Process_Ai2d,
        'web': Process_Web,
        'lec': Process_Lec,
        'office': Process_Office,
        'comics': Process_Comics,
        'nextqa': Process_Nextqa,
        'animals': Process_Animals,
        'med': Process_Med,
        'vasr': Process_Vasr,
        'coco': Process_Coco,
        'resisc': Process_Resisc,
        'ocr': Process_Ocr,
        'magic': Process_MagicBrush,
        'wikiart': Process_Wikiart,
        'food': Process_Food101,
        'sci': Process_ScienceQA,
        'alfred': Process_Alfred,
        'natural': Process_Natural
    }

    # Call the appropriate function if the type is valid
    if type in process_functions:
        process_functions[type]()
    else:
        print(f"Unknown type: {type}")
if __name__ == '__main__':
    Process('natural')




