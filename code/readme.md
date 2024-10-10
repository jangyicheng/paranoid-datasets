# 代码备注

各个函数的参数文档均以代码中注释形式写在代码文件中，这里仅提供一个全局的思路

workflow：

ProcessDataset→utils(save_json)→data_Builder

1.先处理数据集，形成如下格式：

某一类图像的根目录root——子文件夹（编号为1、2、3….)——图片（编号为1、2、3…+后缀名）

具体见multi-img-multi-turn文件夹中示例

2.使用utils.py中save_json函数给对应root生成image_and_texts.json，记录每一组的图片路径和图片备注信息

3.使用data_Builder中generate…answer函数读取image_and_texts.json，生成问答对，以json格式存入每一子文件夹下

---

data_Builder.py为核心代码

`get_response`用于和API交互，输入为system级别prompt，用户级别message，以及输入图片的文件路径。返回API的回复结果，以及耗费token数

`single_turn_generate_questions_and_answers`用于对给定目录对应的image_and_texts.json文件生成问答对，并存入json文件下

每一轮具体流程如下：

1.读取文件，获得图片路径以及其辅助信息

2.提问Agent一次性提出所有问题（如果要求轮次较大，则需多次提问），其中问题前面带有形如[QUESTION_TYPE]的标签表示问题类型，并记录出现过的所有问题类型（防止出现重复）

3.对所有问题按引用图片序号大小进行排序，按序号越大的图片越晚提问的原则（有助于节省token，并符合自然对话规律）

4.回答Agent多轮回复所有问题，每次按标签仅输入问题所涉及到的图片

5.随机抽取一（多）个问答对产生深入问题，并插入到该问答下一位置

6.存储完整对话历史，每一图片输入轮次，花费token数

`multi_turn_generate_questions_and_answers` 基本同上，但由于耗费过大已经废弃

区别在于，是一问一答交替产生

---

ProcessDataset.py
`Process`  选择不同数据集的parquet（或其他文件），进行预处理

得到完整格式的数据，即

某一类图像的根目录root——子文件夹（编号为1、2、3….)——图片（编号为1、2、3…+后缀名）

Process（‘type’）表示预处理type类型的数据集

---

prompt.py
`QuestionPrompt，AnswerPrompt`

分别为提问和回答的agent所需的两种Prompt，还可以根据数据集类别选择对应的prompt

使用方式：

QPrompt=QuestionPrompt()#实例化

prompt=QPrompt.type(’doc’)#选择具体任务类型，这里是doc类型

---

utils.py存放工具函数的文件

重要函数：
`save_to_json`

对于给定目录下指定预处理后的文件夹进行处理

并进行图片缩放、利用chatgpt进行信息增强等操作

得到每一root文件夹所对应的完整格式的图文信息json文件：

1. 遍历给定root文件夹下所有文件夹，对于文件夹遍历其所有图片
2. 若resize参数为True，则对每张图片进行缩放至512*512大小以下
3. 若满足`support_info` 为True，即调用API生成辅助信息txt格式；若`regenerate_depict` 为True，则重新生成覆盖此前生成的辅助信息
4. 将辅助信息和图片路径均添加至json文件中

下面链接给出对于不同数据集其应该使用参数的情况

[](https://www.notion.so/10d7ae070b5a80e19d78e1dce276d27f?pvs=21) 

下面是一些无关紧要的工具函数：
`reorder_images_in_directory`
对给定文件夹下所有图片按1、2、3…+后缀名的格式进行重命名
`rename_folders_in_sequence`

对所有文件夹进行重命名（与缩放图片函数配合使用）
`divide_images_into_subfolders`
将指定目录下所有图片按N张一组划分，分到不同的子文件夹下
`shuffle_folders_in_range`

随机打乱给定root下的所有文件夹，并能够保证文件夹名称仍按顺序排列
`count_images_in_directory`

计算并可视化每一子文件夹中图片数量是否合适，用于检验预处理结果

imgcluster.py,textcluster.py（基本没用这两个文件）

分别是对于给定数据集中图片聚类，和按对应文本进行聚类，并根据聚类结果

生成合适的数据集格式

examine.py
`process_json`    对于  data_Builder生成的conversation.json文件，提取其中问题和待评测LLM交互，生成LLM的对话数据集，以json格式输出