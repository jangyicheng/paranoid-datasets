import base64
import glob
import json
import os
import random
import re
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
import subprocess
import fitz
import requests
from PIL import Image,ImageDraw
import io
import pyarrow.parquet as pq
import pandas as pd
from PyPDF2 import PdfFileReader
from comtypes.client import CreateObject
from datasets import load_dataset
from matplotlib import pyplot as plt
from openai import OpenAI
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pdf2image import convert_from_path
from pptx.util import Inches
from pycocotools.coco import COCO
from tqdm import tqdm
import win32com.client
api_key = ""
organization = ""
client=OpenAI(api_key= "",organization="")

system_prompt = """You are an advanced AI assistant specialized in analyzing and interpreting images. Your task is to provide detailed and insightful descriptions of the images presented to you. Follow these guidelines:

1. Describe the content of each image thoroughly, focusing on relevant details based on the image type.
2. For charts and graphs, identify the type, subject, data characteristics, and provide an in-depth analysis of the data presented.
3. If given a specific question about the image(s), address it comprehensively after providing the general description.
4. Structure your response clearly, using <image-1>, <image-2>, etc., to refer to multiple images in the order they were provided.
5. Ensure your descriptions are objective, accurate, and informative.
6. If applicable, note any text present in the images and its relevance to the overall content.
7. Highlight any unusual or notable aspects of the images that might be of interest.

Respond in a well-organized, coherent manner, separating your analysis for each image while maintaining a cohesive overall response."""

chart_prompt ="""
You will be provided with a text or image containing charts, tables, or datasets. Your task is to extract and analyze the information, and answer the following questions using concise, objective, and accurate language:
1. Information Description and Analysis:  - Describe the main content presented in the chart or table. For example, specify the type of chart, the objects being compared (such as countries, years, categories, etc.), the unit of measurement, and what the X-axis and Y-axis represent.
  - Ensure that your description comprehensively covers the basic information of the chart or table.
2. Number Recognition and Description:  - Accurately identify and describe key data and numerical values present in the chart or table.
  - Highlight specific years, categories, or objects with the highest or lowest data values and provide the corresponding figures.
3. Trend Analysis and Explanation:  - Analyze the trends, changes, or distribution of data in the chart or table.
You  should describe the content of each image thoroughly, focusing on relevant details based on the image type.for charts and graphs, identify the type, subject, data characteristics, and provide an in-depth analysis of the data presented.
"""

doc_prompt ="""
You will be provided with an image containing textual information. Your task is to perform the following analyses and descriptions:
1. Summary of Content:
  - Summarize the content expressed by the text in the image. Identify the type of document and its main information.
2. Text Recognition and Description:
  - Accurately recognize and describe the textual information presented in the image. Be as precise and detailed as possible.
3. Implicit Information:
  - Describe any implicit information suggested by the image.
Requirements:
- Ensure your descriptions are concise, objective, and accurate, focusing on the essential aspects of the textual content,using no more than 300 words!
"""

sci_prompt ="""
You will be provided with an image related to scientific knowledge. Your task is to perform the following analyses and explanations:

1. Content Analysis:  
  - Objectively describe what the image shows, focusing on the scientific content presented. Be as precise and detailed as possible.

2. Scientific Principle Explanation:  
  - Explain and elaborate on the scientific principles involved in the image. Provide clear and concise descriptions of the underlying science principles.

3. Clarity and Accuracy:  
  - Ensure that your analysis and explanations are concise and accurate, highlighting the key aspects of the scientific content.

"""

med_prompt = """
You will be provided with medical images. Your task is to analyze the images as follows:

1. Image Categorization and Content Identification:
   - Analyze the provided medical images to determine the category of each image and specify its content.

2. Disease Diagnosis and Description:
   - Analyze the provided images to identify the medical condition depicted and describe the type of disease present.

Requirements:
- Provide concise, objective, and accurate analyses focusing on categorizing the images, identifying specific content, diagnosing medical conditions, and describing diseases depicted in the images.
"""

scr_prompt ="""
You will be provided with screenshot images. Your task is to analyze the images as follows:

1. Text Content Identification:
   - Extract and provide the specific text content present in the given screenshots.

2. Element and Content Analysis:
   - Analyze the screenshots to identify and describe the elements and main content present in the images.

Requirements:
- Provide concise, objective, and accurate analyses focusing on extracting the text content and identifying the elements and main content depicted in the screenshots.
"""

web_prompt ="""
You will be provided with screenshot images. Your task is to analyze the images as follows:

1. Text Content Identification:
   - Extract and provide the specific text content present in the given screenshots.

2. Element and Content Analysis:
   - Analyze the screenshots to identify and describe the elements and main content present in the images.

Requirements:
- Provide concise, objective, and accurate analyses focusing on extracting the text content and identifying the elements and main content depicted in the screenshots.
"""

lec_prompt =""" You will be provided with images related to NLP, ML, AI, DL, or IR. Your task is to analyze the images as follows:
1. Content Analysis and Explanation:
  - Analyze the images to explain the main content, focusing on the relevant concepts and relationships.
2. Text Extraction:
  - Extract and present all textual information visible in the images.
3. Formula Explanation:
  - Identify and explain any mathematical or computer science formulas present in the images.
Requirements:
- Ensure your descriptions are concise, objective, and accurate, focusing on the essential aspects of the textual content."""

com_prompt=""" Please extract the text from the images. Only return the text content. If there is no text, please return <none>. Format as: text:"Next day... A man and boy stand on a barge in New York Bay..."""""
prompt_type2prompt={"sys":system_prompt,"sci":sci_prompt,"doc":doc_prompt,
                    "chart":chart_prompt,"med":med_prompt,"scr":scr_prompt,
                    'web':web_prompt,'lec':lec_prompt,'com':com_prompt}
#根据数据集类型，选择其对应的增强prompt类型




def analyze_image(image_path, prompt_type = None,specific_question=None):
    def get_response(prompt=" ", model="gpt-4o", max_tokens=1024, temperature=0.2, message=None, images=None,
                     conversation_history=None):
        if message is None:
            message = [{"role": "system", "content": prompt}]

        messages = [{"role": "system", "content": prompt}]

        image_messages = []
        if images:
            for image_url in images:
                with open(image_url, 'rb') as image_file:
                    image_data = image_file.read()
                base64_image = base64.b64encode(image_data).decode('utf-8')
                image_messages.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpg;base64,{base64_image}"}
                })

        if not conversation_history:
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": " prompt:" + message}, *image_messages]})
        else:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Conversation history:{conversation_history}" + " prompt:" + message},
                    *image_messages]})


        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages
        )
        total_tokens = response.usage.total_tokens

        if total_tokens > 10000:
            print("Token limit exceeded.")
            exit(0)
            return None
        print(total_tokens)
        content = response.choices[0].message.content
        return content, total_tokens

    prompt=prompt_type2prompt[prompt_type]#根据prompt_type选择prompt




    if specific_question!=None:
        messages = f"Please analyze the following images and answer this specific question: {specific_question}"
    else:
        messages="Please analyze the following images,make sure your answer is no more than 400 words"

    content, total_tokens = get_response(prompt, message=messages, images=image_path)
    return content, total_tokens

def rename_images_in_directory(root_dir):#将root_dir目录下所有文件夹中图片按1、2、3...+后缀名重命名
    """Remove 'resized_' prefix from image filenames."""
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.json'):
                continue
            file_extension = os.path.splitext(filename)[1]
            new_filename = re.sub(r'^resized_', '', filename)
            numeric_part = re.findall(r'\d+', new_filename)
            if numeric_part:
                new_filename = f"{numeric_part[0]}{file_extension}"
                old_filepath = os.path.join(dirpath, filename)
                new_filepath = os.path.join(dirpath, new_filename)
                if old_filepath != new_filepath:
                    if os.path.exists(new_filepath):
                        print(f'Remove {new_filepath}')
                        os.remove(new_filepath)
                    os.rename(old_filepath, new_filepath)
                    print(f'Renamed: {old_filepath} -> {new_filepath}')


def process_image(file_name, resize):#缩放图片
    """Resize image if necessary."""
    with Image.open(file_name) as img:
        original_width, original_height = img.size
        if (original_width > 512 or original_height > 512) and resize:#将图片缩放到512*512以下，否则会耗费过多token
            ratio = min(512 / original_width, 512 / original_height)
            new_size = (int(original_width * ratio), int(original_height * ratio))
            img = img.resize(new_size)
            temp_path = os.path.join(os.path.dirname(file_name), f"resized_{os.path.basename(file_name)}")
            img.save(temp_path)
            return temp_path
    return file_name


def analyze_and_label(file_name, txt_file_name, prompt_type, support_info, regenerate_depict):
    """Analyze image and generate or reuse text label."""
    if os.path.exists(txt_file_name):#若已经存在描述
        if not regenerate_depict:#且不需要重新生成
            with open(txt_file_name, 'r', encoding='utf-8') as txt_file:
                return txt_file.read()
        else:#需要重新生成
            result, _ = analyze_image([file_name], prompt_type)  #
            with open(txt_file_name, 'w', encoding='utf-8') as txt_file:
                txt_file.write(result)
            return result
    #不存在描述
    if support_info:#若重新生成描述
        result, _ = analyze_image([file_name], prompt_type)  #
        with open(txt_file_name, 'w', encoding='utf-8') as txt_file:
            txt_file.write(result)
        return result
    return ""

#核心代码
def save_to_json(image_file, file_num, output_file='images_and_texts.json', resize=True, support_info=False,
                 prompt_type="sys", regenerate_depict=False):
    """
    保存图像和文本信息到JSON文件。

    参数：
    - image_file (str): 包含图像文件的目录。
    - file_num (int): 文件夹数量。
    - output_file (str): 输出JSON文件的名称。默认值为'images_and_texts.json'。
    - resize (bool): 是否调整图像大小。默认值为True。
    - support_info (bool): 是否包括添加辅助信息。默认值为False。
    - prompt_type (str): 提示类型。默认值为"sys"。
    - regenerate_depict (bool): 是否重新生成图像描述。默认值为False。
    """
    image_path = os.path.join(os.getcwd(), image_file)
    if not os.path.exists(image_path):
        print("Image file directory does not exist!")
        return None

    images_and_texts = []
    output_file = os.path.join(image_path, output_file)

    for idx in tqdm(range(file_num)):
        folder_path = os.path.join(image_file, str(idx + 1))
        if not os.path.exists(folder_path):
            print("Folder path does not exist!")
            return None

        image_paths = []
        texts = []
        file_count = 0

        for file_name in sorted(glob.glob(os.path.join(folder_path, '*'))):
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')):
                file_count += 1
                processed_file_name = process_image(file_name, resize)#缩放图片，减小token数

                if processed_file_name != file_name:
                    print(f"Remove {file_name}")
                    os.remove(file_name)


        rename_images_in_directory(image_file)#将缩放后的图片重新命名回原名
        for file_name in sorted(glob.glob(os.path.join(folder_path, '*'))):
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')):
                image_paths.append(file_name)
                txt_file_name = os.path.splitext(file_name)[0] + '.txt'
                discription = analyze_and_label(file_name, txt_file_name, prompt_type, support_info, regenerate_depict)
                #根据要求对图片进行AI提取信息，加入texts中
                texts.append(f"<image-{file_count}>:{discription}")

        images_and_texts.append({
            "images": image_paths,
            "texts": texts
        })

    with open(output_file, 'w') as f:
        json.dump(images_and_texts, f, indent=4)
    print(f"Saved data to {output_file}")



def is_image_file(filename):#判断指定文件是否属于图片类型
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'))

def reorder_images_in_directory(root_dir):#将指定目录下所有文件夹重新命名为1、2、3...后缀不变
    # 遍历当前目录下的每一组叶子文件夹
    for subdir, _, files in os.walk(root_dir):
        image_files = [f for f in files if is_image_file(f)]

        if not image_files:
            continue

        # 检查是否所有图片文件名为按照1-n的数字排列
        numeric_filenames = [int(re.match(r'(\d+)', f).group(1)) for f in image_files if re.match(r'^\d+\.', f)]

        # 如果文件名是按1-n排列的数字，则跳过这个文件夹
        if sorted(numeric_filenames) == list(range(1, len(image_files) + 1)):
            continue

        # 否则将图片重新命名为1.jpg, 2.jpg, ..., n.jpg
        for i, filename in enumerate(sorted(image_files), 1):
            file_ext = os.path.splitext(filename)[1]
            new_name = f"{i}{file_ext}"
            old_path = os.path.join(subdir, filename)
            new_path = os.path.join(subdir, new_name)
            os.rename(old_path, new_path)
            print(f"Renamed {old_path} to {new_path}")

def rename_folders_in_sequence(folder_path):#将所有文件夹重新命名为1、2、3......
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

def divide_images_into_subfolders(source_folder, batch_size=1000):
    """
    将指定文件夹下所有图片按每组1000张划分，存入子文件夹中。
    （之前用于聚类操作）
    :param source_folder: 源文件夹路径
    :param batch_size: 每个子文件夹的图片数量，默认1000张
    """
    source_path = Path(source_folder)
    if not source_path.is_dir():
        raise NotADirectoryError(f"{source_folder} is not a valid directory.")

    # 获取所有图片文件
    image_files = [file for file in source_path.iterdir() if
                   file.is_file() and is_image_file(file.suffix)]

    # 计算需要的子文件夹数量
    total_images = len(image_files)
    num_folders = (total_images + batch_size - 1) // batch_size

    # 创建并移动图片到子文件夹中
    for i in range(num_folders):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, total_images)

        subfolder_name = source_path / f"batch_{i + 1}"
        subfolder_name.mkdir(exist_ok=True)

        for image_file in image_files[start_index:end_index]:
            shutil.move(str(image_file), str(subfolder_name / image_file.name))

    print(f"Successfully divided {total_images} images into {num_folders} subfolders.")


#历史遗留问题
def extract_and_save_geo_conversations(json_file_path="D:/datasets/GEO170K/images/qa_tuning.json", output_base_dir="D:/datasets/GEO170K/images/train/JSON"):
    def merge_json_files_in_subfolders(base_dir):
        # 遍历 base_dir 下的所有子文件夹
        for subfolder_name in os.listdir(base_dir):
            subfolder_path = os.path.join(base_dir, subfolder_name)

            # 确保是一个目录
            if os.path.isdir(subfolder_path):
                # 用于存储合并后的数据
                merged_data = []

                # 遍历子文件夹中的所有 JSON 文件
                for file_name in os.listdir(subfolder_path):
                    file_path = os.path.join(subfolder_path, file_name)

                    # 确保是一个 JSON 文件
                    if os.path.isfile(file_path) and file_name.endswith('.json'):
                        with open(file_path, 'r', encoding='utf-8') as json_file:
                            data = json.load(json_file)
                            # 合并数据，假设每个 JSON 文件的数据是一个列表
                            if isinstance(data, list):
                                merged_data.append([data])
                            else:
                                # 如果数据不是列表，可以根据需要进行处理
                                print(
                                    f"Warning: JSON file {file_name} in folder {subfolder_name} does not contain a list.")

                # 将合并后的数据保存到 merged.json 文件
                merged_json_path = os.path.join(subfolder_path, 'merged.json')
                with open(merged_json_path, 'w', encoding='utf-8') as merged_file:
                    json.dump(merged_data, merged_file, ensure_ascii=False, indent=4)

                print(f"Successfully merged JSON files in folder: {subfolder_name}")
    # 读取JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 遍历每个条目
    for entry in data:
        image_path = entry["image"]
        conversations = entry["conversations"]

        # 获取图片文件名（不带扩展名）
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        # 创建对应的文件夹
        image_folder = os.path.join(output_base_dir, image_name)
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)

        # 获取文件夹下的现有文件数量
        existing_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
        idx = len(existing_files) + 1

        if idx <= 8:
            # 保存问答对到json文件
            json_filename = os.path.join(image_folder, f"{idx}.json")
            with open(json_filename, 'w', encoding='utf-8') as json_file:
                json.dump(conversations, json_file, ensure_ascii=False, indent=4)
    merge_json_files_in_subfolders(output_base_dir)



def filter_directories(root_dir):#删除少于2张图片的文件夹
    # 遍历目录
    for root, dirs, files in os.walk(root_dir, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            image_files = [f for f in os.listdir(dir_path) if is_image_file(os.path.join(dir_path, f))]

            # 如果图片数量少于2张，删除文件夹
            if len(image_files) < 2:
                shutil.rmtree(dir_path)
                print(f"Deleted directory: {dir_path}")

    # 删除之后对所有文件夹重命名
    rename_folders_in_sequence(root_dir)



def shuffle_folders_in_range(directory, start_num, end_num):
    """
    在指定范围内对文件夹进行随机重排。

    Args:
        directory (str): 包含文件夹的目录路径
        start_num (int): 起始数字
        end_num (int): 结束数字

    Returns:
        None
    """
    # 获取目录中的所有文件夹
    folders = [folder for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder))]

    # 筛选出在指定范围内的文件夹
    folders_to_shuffle = [str(i) for i in range(start_num, end_num + 1) if str(i) in folders]

    # 随机重排文件夹列表
    random.shuffle(folders_to_shuffle)

    # 重命名文件夹以便重新排列它们
    for i, folder_name in enumerate(folders_to_shuffle):
        old_folder = os.path.join(directory, folder_name)
        tmp_folder = os.path.join(directory, f".tmp_folder_{i}")
        os.rename(old_folder, tmp_folder)

    # 将临时重命名的文件夹恢复为原始名称
    for i, folder_name in enumerate(folders_to_shuffle):
        tmp_folder = os.path.join(directory, f".tmp_folder_{i}")
        new_folder = os.path.join(directory, folder_name)
        os.rename(tmp_folder, new_folder)


def count_images_in_directory(directory):
    """
    计算指定目录中每个文件夹中的图像数量，并可视化分布。

    参数:
    directory (str): 包含图像文件夹的目录路径。

    返回:
    None

    该函数遍历目录结构，计算每个文件夹中的图像数量，并创建可视化图表显示基于图像数量的文件夹分布。

    同时打印出主目录中包含少于2个图像的子目录。
    """
    def visualize_image_counts(image_counts, directory):
        """
               可视化文件夹中图像数量的分布。

               参数:
               image_counts (dict): 包含每个文件夹中图像数量的字典。
               directory (str): 正在分析的目录路径。

               返回:
               None

               该函数创建一个条形图，显示不同图像数量的文件夹数量。
               """
        counts = list(image_counts.keys())
        folder_counts = list(image_counts.values())

        plt.figure(figsize=(10, 6))
        bars = plt.bar(counts, folder_counts, color='skyblue')
        plt.xlabel('Number of Images in Folder')
        plt.ylabel('Number of Folders')
        plt.title(directory)
        plt.xticks(rotation=90)  # Rotate x-axis labels if necessary
        plt.tight_layout()

        # Add text labels on top of the bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), va='bottom')

        plt.show()

    image_counts = defaultdict(int)

    for root, dirs, files in os.walk(directory):
        image_count = 0
        for file in files:
            file_path = os.path.join(root, file)
            if is_image_file(file_path):
                image_count += 1
        image_counts[image_count] += 1
    image_counts[0]-=1
    visualize_image_counts(image_counts,directory)

    # Print folders with fewer than 2 images in each subdirectory
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path):
            for root, dirs, files in os.walk(subdir_path):
                image_count = 0
                for file in files:
                    file_path = os.path.join(root, file)
                    if is_image_file(file_path):
                        image_count += 1
                if image_count < 2:
                    print(root)


def json_to_txt(json_file_path, txt_file_path):
        # 读取JSON文件
        with open(json_file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)

        # 将JSON数据写入TXT文件
        with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(json.dumps(data, indent=4, ensure_ascii=False))

def txt_to_json(txt_file_path, json_file_path):
        # 读取TXT文件
        with open(txt_file_path, 'r', encoding='utf-8') as txt_file:
            data = txt_file.read()

        # 将TXT数据解析为JSON
        json_data = json.loads(data)

        # 将JSON数据写入JSON文件
        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(json_data, json_file, indent=4, ensure_ascii=False)

def valid_location():
    try:
        response = requests.get('https://ipinfo.io/json')
        data = response.json()
        if 'country' in data:
            if data['country'] == 'CN' or data['country'] == 'HK':
                print("在中国境内")
                return False
            else:
                #print("不在中国境内")
                return True
        else:
            print("无法获取国家信息")
            return False
    except Exception as e:
        print(f"获取地理位置信息时出错: {e}")
        return False

if __name__ == '__main__':


    focus_dataset=['VASR',  'ANIMALS', 'OFFICEHOME',  'RESISC', "OCR", 'COCO','Wikiart','MAGICBRUSH','FOOD','COMICS'
                   'NEXTQA','SCIENCEQA','PubMed',"CHART_QA" ,'AI2D','DOCVQA','SCREENQA','MIND2WEB','LECTUREBANK',
                   ]
    prompt_type = ['VASR',  'ANIMALS', 'OFFICEHOME', 'RESISC', "OCR", 'COCO','ART','MAGIC',"FOOD",'COMICS','NEXTQA',
                   'SCIENCEQA','PubMed','chart','sci','doc','scr','web','lec']
    #对应增强信息所用的prompt类型




    #可以按需一次处理一部分dataset的json文件如
    focus_dataset = ['VASR', 'ANIMALS', 'OFFICEHOME']
    prompt_type = ['VASR', 'ANIMALS', 'OFFICEHOME']

    for id,data in enumerate(focus_dataset):
          save_to_json(data, 10, resize=True,support_info=False,prompt_type=prompt_type[id],
                    regenerate_depict=False)
    #处理前5组图片，resize是是否缩放图片（最大长宽为512），对于需要高分辨率的图片，不可以缩放，





























