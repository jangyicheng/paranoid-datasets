import base64
import os
import random
import re
import sys
import time
import json
import requests
from openai import OpenAI
from tqdm import tqdm
from prompt import QuestionPrompt, AnswerPrompt,EvaluatePrompt

#初始化API
api_key = ""
organization = ""
client=OpenAI(api_key= "",organization="")


"""
使用API和gpt-4o进行交互，
prompt：str 系统级prompt，用于设定agent的角色
message：str 每次调用时的具体指令
images:list[str] 传入图片路径的列表
conversation_history：list  对话历史的列表
"""
def get_response(prompt=" ", model="gpt-4o", max_tokens=1024, temperature=0.2, message=None, images=None,conversation_history=None):

    messages=[{"role": "system", "content": prompt}]


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
                    "content": [{"type": "text", "text": " prompt:"+ message},*image_messages]})
    else:
        messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": f"Conversation history:{conversation_history}"+" prompt:"+message},
                        *image_messages]})

    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=messages
    )
    total_tokens = response.usage.total_tokens
    content = response.choices[0].message.content
    return content, total_tokens


'''
多轮次生成问答对，一问一答交替进行，但是后来发现这样效率低下且耗费tokens，已经废弃
images_and_texts：解析后的json格式数据
total_turns：问答轮次数
type：dataset名称，参见prompt.py文件，用于生成某些指定数据集对应任务
m,n分别表示从给定文件夹下第m到第n个子文件进行遍历生成问答
'''
def multi_turn_generate_questions_and_answers(images_and_texts, total_turns, type, m=0, n=10000000):
    conversation_id = 1
    total_tokens_used =0
    temp_tokens =0
    token_threshold =200000
    valid_location()#检查API调用是否安全
    Qprompt = QuestionPrompt()#prompt实例化
    Aprompt = AnswerPrompt()
    images_and_texts=images_and_texts[m:n]
    for item in tqdm(images_and_texts, desc="Processing images and texts"):
        images = item['images']
        texts = item['texts']
        turns =item['turns']
        conversation_history = []
        questions=[]
        answers=[]
        n=len(images)
        whole_turns =random.randint(0,total_turns) if n<=4 else -1#该轮次一次对所有图像进行提问
        #indepth_turns_num=random.randint(1,2)
        indepth_turns_num=1#表示生成深入问题的数量
        indepth_turns = random.sample(range(1, total_turns), indepth_turns_num)
        past_question_type = ['IN-DEPTH QUESTIONS']#用于记录所有已经在对话历史中出现过的问题类型
        pattern = r'^\s*(\[([\s\S]+?)\])'#用于提取问题类型
        for turn in range(total_turns):
            # 构建提问GPT的输入
            if(turn >=1 and turn in indepth_turns):#提问关于上一问答的深入问题
                input_prompt = f"Please ask a in-depth question in response to the other person's last round of responses!You can quote the other person's reply as appropriate and ask more in-depth questions about the reply.The question should begin with Q: [IN-DEPTH QUESTIONS],Please keep your questions as concise as possible"
            elif(turn == whole_turns):#该轮次一次对所有图像进行提问
                input_prompt="Please design a valuable question about all the images at the same time"
            else:
                input_prompt=f"Please design a new question that doesn't come from one of the categories below:{past_question_type}!The question should begin with Q: [QUESTION TYPE],for example,Q:[SCENE UNDERSTANDING]"

            input_prompt += f""" Text:  {[texts[i] for i in range(n) if turns[i] <= turn]}\n"""
            input_images = [images[i] for i in range(n) if turns[i] <= turn]#每次仅输入满足条件的图片
            questions_response, tokens = get_response(Qprompt.prompt(type), message=
            input_prompt, images=input_images, temperature=0.8,conversation_history=conversation_history)

            questions_temp = questions_response.split("Q:")  # 假设问题按行分隔
            total_tokens_used += tokens
            temp_tokens += tokens
            # 构建回答GPT的输入
            # 遍历直到找到一个未在历史对话中出现过的问题
            question = ""
            while not question or any(question.strip() in message["content"] for message in conversation_history):
                question = random.choice(questions_temp[1:]) if questions_temp else "Q: What is shown in the images?"
            matches = re.search(pattern, question)
            past_question_type.append(matches.group(1))#添加到历史问题类型中
            questions.append(question)

            #--------------------------------------------------------------------------
            #回答阶段
            input_prompt=question+"Please limit your answer to about 150 words."
            input_prompt += f"""Text:  {[texts[i] for i in range(n) if turns[i] <= turn]}\n"""
            answer, tokens = get_response(Aprompt.prompt(type),
            message=input_prompt, images=input_images ,conversation_history=conversation_history)


            answers.append(answer)
            total_tokens_used += tokens
            temp_tokens+= tokens
            # 检查是否超过20万个tokens
            if temp_tokens >= token_threshold:
                print(f"Reached {token_threshold} tokens. Sleeping for 100 seconds...")
                time.sleep(120)
                temp_tokens = 0  # 重置token计数器

            # 更新对话历史
            conversation_history.append({"role": "user", "content": question})
            conversation_history.append({"role": "assistant", "content": answer})
            #上下文仅保留n轮对话
            # if len(conversation_history) > 6:
            #     conversation_history = conversation_history[-6:]

            # 保存对话到JSON文件
            data = {
                "images": images,
                "question": questions,
                "answer": answers,
                "conversation_history": conversation_history,
                'input_turns':turns
            }

            print(f"Total tokens used: {total_tokens_used}")
            output_file = os.path.join(os.path.dirname(str(images[0])), "conversation.json")
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=4)
        conversation_id += 1
    print(f"Total tokens used: {total_tokens_used}")
    print(f"任务已经处理完毕！")



'''
单轮次生成所有问题，后续依次回答所有问题，in-depth问题单独添加
images_and_texts：解析后的json格式数据
total_turns：问答轮次数
type：dataset名称，参见prompt.py文件，用于生成某些指定数据集对应任务
m,n分别表示从给定文件夹下第m到第n个子文件进行遍历生成问答
regenerate：bool 表示是否重新覆盖原有的问答对

对于该函数与上面函数相同部分不添加注释了，参见上文
'''
def single_turn_generate_questions_and_answers(images_and_texts, total_turns, type, m=0, n=10000000, regenerate=False):
    def extract_max_image_index(question):
        # 提取 question 中的 <image-x> 标签
        image_indices = re.findall(r'<image-(\d+)>', question)
        # 将提取到的索引转换为整数并找到最大值
        if image_indices:
            return max(map(int, image_indices))
        return -1  # 如果没有找到 <image-x> 标签，返回 -1

    conversation_id = 1
    total_tokens_used = 0
    token_threshold = 200000
    valid_location()
    Qprompt = QuestionPrompt()
    Aprompt = AnswerPrompt()
    images_and_texts = images_and_texts[m:n]

    for item in tqdm(images_and_texts, desc="Processing images and texts"):
        images = item['images']
        texts = item['texts']
        turns = [-1]*len(images)
        conversation_history = []
        questions = []
        answers = []
        total_turns = random.randint(5,8)
        total_turns = total_turns
        n = len(images)
        past_question_type = ['IN-DEPTH QUESTIONS']
        pattern = r'^\s*(\[([\s\S]+?)\])'

        # Check if conversation.json exists in the folder containing the images
        image_folder = os.path.dirname(images[0])
        conversation_file_path = os.path.join(image_folder, 'conversation.json')

        if not regenerate and os.path.exists(conversation_file_path):#判断是否重新生成
            print(f"Skipping {image_folder} as conversation.json already exists.")
            continue

        # Initialize tokens_used for the current group of images
        tokens_used = 0

        # Step 1: Generate total_turns-1 unique questions
        #这里目前有个潜在bug，对于轮次太大的情况，问题顺序没有很好的排序，不过很容易改正
        while len(questions) < total_turns - 1:
            input_prompt = f"Please design some new questions that doesn't come from one of the categories below:{past_question_type}!The question should begin with Q: [QUESTION TYPE],for example,Q:[SCENE UNDERSTANDING]"
            input_prompt += f"""Text descriptions:  {[texts[i] for i in range(n)]}\n"""
            input_images = [images[i] for i in range(n)]

            questions_response, tokens = get_response(Qprompt.prompt(type), message=
                input_prompt, images=input_images, temperature=0.8,conversation_history=conversation_history)
            questions_temp = questions_response.split("Q:")  # 假设问题按行分隔
            total_tokens_used += tokens
            tokens_used += tokens
            sorted_questions = sorted(questions_temp, key=extract_max_image_index)
            '''
            按问题中提及最大的图片索引进行降序排序（为使得总输入的图片次数最少
            如q1包含<image-1>,<image-5>
              q2包含<image-2>,<image-4>
              则由于4<5，故q2应该出现在q1前面
            '''

            for turn,question in enumerate(sorted_questions[1:-1]):
                matches = re.search(pattern, question)
                if matches and matches.group(1) not in past_question_type:
                    past_question_type.append(matches.group(1))
                    questions.append(question.strip())
                    if len(questions) >= total_turns - 2:
                        break

            matches = re.search(pattern, sorted_questions[-1])#为保证尽可能覆盖所有图片，强制选择最后一个问题（即覆盖最后一个图片
            if matches and matches.group(1) not in past_question_type:
                past_question_type.append(matches.group(1))
                questions.append(sorted_questions[-1].strip())

            if tokens_used >= token_threshold:
                print(f"Reached {token_threshold} tokens. Sleeping for 100 seconds...")
                time.sleep(120)
                tokens_used = 0  # 重置token计数器


        # Step 2: Get answers for the generated questions
        for question in questions:
            input_images = []

            # 提取 question 中的 <image-x> 标签
            image_indices = re.findall(r'<image-(\d+)>', question)

            # 将提取的图像索引转换为整数
            image_indices = sorted([int(index) - 1 for index in image_indices])

            # 去重并保持索引顺序
            image_indices = list(dict.fromkeys(image_indices))


            # 从 images 列表中提取相应的图像
            question_images = [images[i] for i in image_indices]


            # 将 question_images 添加到 input_images
            input_images.extend(image for image in question_images if image not in input_images)

            # 对 input_images 按文件名中的索引排序
            input_images.sort(key=lambda x: int(x.split('\\')[-1].split('.')[0]))


            input_prompt = question + " Please limit your answer to about 150 words."
            input_prompt += f"""Text descriptions:  {[texts[i] for i in range(n) if turns[i] <= total_turns]}\n"""
            answer, tokens = get_response(Aprompt.prompt(type),
            message=input_prompt, images=input_images,conversation_history=conversation_history)

            answers.append(answer)
            total_tokens_used += tokens
            tokens_used += tokens

            conversation_history.append({"role": "user", "content": question})
            conversation_history.append({"role": "assistant", "content": answer})

            if tokens_used >= token_threshold:
                print(f"Reached {token_threshold} tokens. Sleeping for 100 seconds...")
                time.sleep(120)
                tokens_used = 0  # 重置token计数器

        # Step 3: Generate an in-depth question and answer
        random_index = random.randint(1, len(questions) - 1)#随机轮次插入深入问题
        original_question = questions[random_index]#获取关于深入问题的原问答对
        original_answer = answers[random_index]

        input_prompt = f"Please ask a in-depth question in response to the other person's last round of responses! You can quote the other person's reply as appropriate and ask more in-depth questions about the reply. The question should begin with Q: [IN-DEPTH QUESTIONS], Please keep your questions as concise as possible"
        input_prompt += f"""Text descriptions:  {[texts[i] for i in range(n) if turns[i] <= total_turns]}\n"""
        input_images = [images[i] for i in range(n) if turns[i] <= total_turns]

        in_depth_question_response, tokens = get_response(Qprompt.prompt(type), message=input_prompt
        , images=input_images, temperature=0.8,conversation_history=[
            {"role": "user", "content": original_question},
            {"role": "assistant", "content": original_answer}])
        in_depth_question = in_depth_question_response.split("Q:")[1].strip()
        total_tokens_used += tokens
        tokens_used += tokens

        input_prompt = in_depth_question + " Please limit your answer to about 150 words."
        input_prompt += f"""Text descriptions:  {[texts[i] for i in range(n) if turns[i] <= total_turns]}\n"""
        in_depth_answer, tokens = get_response(Aprompt.prompt(type), message=
            input_prompt, images=input_images,conversation_history=conversation_history)

        total_tokens_used += tokens
        tokens_used += tokens

        # Insert in-depth Q&A into conversation history
        conversation_history.insert(2 * random_index + 2, {"role": "user", "content": in_depth_question})
        conversation_history.insert(2 * random_index + 3, {"role": "assistant", "content": in_depth_answer})

        if tokens_used >= token_threshold:
            print(f"Reached {token_threshold} tokens. Sleeping for 100 seconds...")
            time.sleep(120)
            tokens_used = 0  # 重置token计数器
        questions.insert(random_index + 1, in_depth_question)
        answers.insert(random_index + 1, in_depth_answer)
        #所有问答对已经生成完毕



        #在最终输出文件中用turns列表记录每一图片最早出现的轮次
        for turn, question in enumerate(questions):
            image_indices = re.findall(r'<image-(\d+)>', question)
            for indice in image_indices:
                if turns[int(indice)-1] == -1:#若没有出现过则初始化为-1
                    turns[int(indice)-1] = turn

        #注意，由于图片索引严格按顺序排列，故不能出现中断情况，turns应该呈现严格递增顺序
        for i in range(len(turns) - 1, 0, -1):
            if turns[i-1] > turns[i]:
                turns[i-1] = turns[i]



        # Save conversation to JSON file
        data = {
            "images": images,
            "question": questions,
            "answer": answers,
            "conversation_history": conversation_history,
            'input_turns': turns,
            'tokens_used': tokens_used  # 记录当前组图片所耗费的token总数
        }

        output_file = os.path.join(os.path.dirname(str(images[0])), "conversation.json")
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=4)

        conversation_id += 1

    print(f"Total tokens used: {total_tokens_used}")
    print(f"任务已经处理完毕！")


# 任务3：评价
# 这里用的是来源于MMDU的评价prompt
# 注意需要补充原始benchmark的问答对作为标答，但是还未实现
def evaluate_answers(images_and_texts, data_file="data",output_file_prefix="evaluation"):
    total_tokens_used=0
    conversation_id=1
    Eprompt=EvaluatePrompt()
    for item in tqdm(images_and_texts, desc="Processing evaluation"):
        image_paths=item['images']
        for file_path in image_paths:
            print(file_path)
            # 使用正则表达式提取文件夹序号
            match = re.search(fr'{data_file}\\(\d+)\\', file_path)
            if match:
                conversation_id = match.group(1)
            else:
                print("未找到匹配的文件夹序号")

        input_file = f"{data_file}/{conversation_id}/conversation.json"
        output_file = f"{data_file}/{conversation_id}/{output_file_prefix}_conversation.json"

        with open(input_file, 'r') as f:
            data = json.load(f)

        images = data['images']
        questions = data['question']
        answers = data['answer']
        conversation_history = data['conversation_history']
        evaluations = []

        for i in range(len(questions)):
            question = questions[i]
            answer = answers[i]
            input_images = [images[m] for m in range(i)]
            # 构建评价GPT的输入
            input_prompt = f"Evaluate the following question and answer:\n\nQuestion: {question}\nAnswer: {answer}"

            evaluation , tokens = get_response(Eprompt.prompt(), message=
            input_prompt, images=input_images)

            total_tokens_used += tokens
            evaluations.append({
                "question": question,
                "answer": answer,
                "evaluation": evaluation
            })

        evaluated_data = {
            "images": images,
            "qa_evaluations": evaluations,
            "conversation_history": conversation_history
        }
        print(output_file)
        with open(output_file, 'w') as f:
            json.dump(evaluated_data, f, indent=4)
    print(f"Total tokens used: {total_tokens_used}")



#生成GEO数据集的问答对
def MATH_QA_pair(images_and_texts,turns=3):
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
    def qa_to_qa(json_data):
        # 提取问题和答案
        question_data = json_data[0]['value']
        answer_data = json_data[1]['value']

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
    def extract_qa(text):
        # 使用正则表达式匹配问题和答案
        question_match = re.search(r'\$\$Q:"(.*?)"', text, re.DOTALL)
        answer_match = re.search(r'\$\$A:"(.*?)"', text, re.DOTALL)

        # 提取匹配到的内容
        question = question_match.group(1).strip() if question_match else None
        answer = answer_match.group(1).strip() if answer_match else None

        return question, answer

    conversation_id = 1
    total_tokens_used =0
    temp_tokens =0
    token_threshold =200000
    valid_location()
    Qprompt = QuestionPrompt()
    Aprompt = AnswerPrompt()

    for item in tqdm(images_and_texts, desc="Processing images and texts"):
        images = item['images']
        temp_texts = item['texts']
        img_turns = item['turns']
        texts=[]
        for text in temp_texts:
            # 提取 JSON 文件路径
            json_path = text.split(":")[1].strip().strip("'")
            # 读取 JSON 文件内容
            with open(json_path, 'r', encoding='utf-8') as file:
                json_content = json.load(file)
                texts.append(json_content)
        conversation_history = []
        questions=[]
        answers=[]
        for idx,qa_pairs in enumerate(texts):
            question_and_answer=qa_to_qa(qa_pairs)
            question ,answer =extract_qa(question_and_answer)
            questions.append(f'for <image-{idx}>:'+question)
            answers.append(f'for <image-{idx}>:'+answer)
            conversation_history.append({"role": "user", "content": f'for <image-{idx}>:'+question})
            conversation_history.append({"role": "assistant", "content": f'for <image-{idx}>:'+answer})
        #以上为前置问答对
        n=len(images)
        past_question_type = ['IN-DEPTH QUESTIONS']
        pattern = r'^\s*(\[([\s\S]+?)\])'
        for turn in range(turns):
            # 构建提问GPT的输入
            if turn == 0:
                input_prompt = "Ask a question about the similarities and differences between the solutions of two maths problems"
            elif turn == 1:
                input_prompt = "Please ask a question about two maths problems examining the relationship between knowledge points"
            else:
                input_prompt= f"Please design a new question that doesn't come from one of the categories below:{past_question_type}!The question should begin with Q: [QUESTION TYPE],for example,Q:[SCENE UNDERSTANDING]"

            input_prompt +=  "the auxiliary information is :"
            input_prompt += f"""Text:  {[texts[i] for i in range(n) if img_turns[i] <= turn]}\n"""
            input_images = [images[i] for i in range(n) if img_turns[i] <= turn]
            questions_response, tokens = get_response(Qprompt.prompt("geo"), message=
            input_prompt, images=input_images, temperature=0.8,conversation_history=conversation_history)
            questions_response=bytes(questions_response, 'utf-8').decode('unicode_escape')#防止出现乱码
            questions_temp = questions_response.split("Q:")  # 假设问题按行分隔

            total_tokens_used += tokens
            temp_tokens += tokens
            # 构建回答GPT的输入
            # 遍历直到找到一个未在历史对话中出现过的问题
            question = ""
            while not question or any(question.strip() in message["content"] for message in conversation_history):
                question = random.choice(questions_temp[1:]) if questions_temp else "Q: What is shown in the images?"
            matches = re.search(pattern, question)
            past_question_type.append(matches.group(1))
            questions.append(question)

            input_prompt=question+"Please limit your answer to about 100 words."
            input_prompt += f"""Text:  {[texts[i] for i in range(n) if img_turns[i] <= turn]}\n"""
            answer , tokens = get_response(Aprompt.prompt("geo"), message=
            input_prompt, images=input_images, conversation_history=conversation_history)
            answer = bytes(answer, 'utf-8').decode('unicode_escape')

            answers.append(answer)
            total_tokens_used += tokens
            temp_tokens+= tokens
            # 检查是否超过20万个tokens
            if temp_tokens >= token_threshold:
                print(f"Reached {token_threshold} tokens. Sleeping for 100 seconds...")
                time.sleep(120)
                temp_tokens = 0  # 重置token计数器

            # 更新对话历史
            conversation_history.append({"role": "user", "content": question})
            conversation_history.append({"role": "assistant", "content": answer})

            # 保存对话到JSON文件
            data = {
                "images": images,
                "question": questions,
                "answer": answers,
                "conversation_history": conversation_history
            }

            print(f"Total tokens used: {total_tokens_used}")
            output_file = os.path.join(os.path.dirname(str(images[0])), "conversation.json")
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=4)
        conversation_id += 1
    print(f"Total tokens used: {total_tokens_used}")
    print(f"任务已经处理完毕！")

'''
用于判断API调用是否安全，若在中国境内则终止程序
'''
def valid_location():
    try:
        response = requests.get('https://ipinfo.io/json')#获取位置
        data = response.json()
        if 'country' in data:
            if data['country'] == 'CN' or data['country'] == 'HK':
                print("在中国境内")
                sys.exit("The VPN you use must be out of China!")
            else:
                print(f"不在中国境内，地区码为{data['country']}")
                return True
        else:
            print("无法获取国家信息")
            return False
    except Exception as e:
        print(f"获取地理位置信息时出错: {e}")
        sys.exit("The VPN you use must be out of China!")
        return False


if __name__ == "__main__":

    # 执行任务1：生成问题
    file_path = 'images_and_texts.json'  # 替换为正确的文件路径
    turns = 8


    my_dataset=['VASR',  'ANIMALS', 'OFFICEHOME',  'RESISC', "OCR", 'COCO','Wikiart','MAGICBRUSH',
                   'FOOD','NEXTQA',"CHART_QA" ,'AI2D','DOCVQA','PubMed','SCREENQA','MIND2WEB','LECTUREBANK',
                   'SCIENCEQA','COMICS']
    for id,dataset in enumerate(my_dataset):#将你已经处理好json文件的文件夹放在这里
        image_path = os.path.join(os.getcwd(), dataset, file_path)
        with open(image_path, 'r') as file:
            images_and_texts = json.load(file)
        single_turn_generate_questions_and_answers(images_and_texts, total_turns=turns,type=dataset,regenerate=True)
        #MATH_QA_pair(images_and_texts,turns=3)
        time.sleep(120)     #避免使用APi频率过高











