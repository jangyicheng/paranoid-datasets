import base64
import json
import os

from openai import OpenAI
from tqdm import tqdm

api_key = "sk-rPuX_cf1mM8Df4qFNadMucBw22Z8aayL-IAobSKjozT3BlbkFJ-iwRq_-C8bZP2CmIA9Psqb6m1QPsYTnz5eJEu_rLEA"
organization = "org-sGcdHuDO64yNd18z2C65ykxE"
client=OpenAI(api_key= "sk-rPuX_cf1mM8Df4qFNadMucBw22Z8aayL-IAobSKjozT3BlbkFJ-iwRq_-C8bZP2CmIA9Psqb6m1QPsYTnz5eJEu_rLEA",organization="org-sGcdHuDO64yNd18z2C65ykxE")



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

    content = response.choices[0].message.content
    return content, total_tokens

"""
根据所给问题对LLM进行测试
"""
def process_json(json_file):
    # 获取文件夹名
    folder_name = os.path.basename(os.path.dirname(json_file))

    with open(json_file, 'r') as file:
        data = json.load(file)

    images = data['images']
    questions = data['question']
    input_turns = data['input_turns']
    conversation_history = []
    total_tokens_used = 0
    image_indices = []
    prompt="Please answer the provided question as concisely and accurately as possible:"
    for i, question in tqdm(enumerate(questions)):
        turn_images = [images[j] for j in range(len(images)) if input_turns[j] <= i]
        response, tokens_used = get_response(
            prompt=prompt,
            model="gpt-4o",
            max_tokens=1024,
            temperature=0.2,
            message=question,
            images=turn_images,
            conversation_history=conversation_history
        )

        total_tokens_used += tokens_used
        turn_image_indices = [j for j in range(len(images)) if input_turns[j] <= i]
        image_indices.append(turn_image_indices)#记录每一轮次输入图片的索引

        print(f"Response for question {i + 1}: {response}")
        conversation_history.append({
            "role": "user",
            "content": question,
        })
        conversation_history.append({
            "role": "assistant",
            "content": response,
        })

    print(f"Total tokens used: {total_tokens_used}")

    # 将对话历史记录、总token数和图片索引分别保存到 examine 文件夹中的同一个文件中
    output_folder = 'examine'
    os.makedirs(output_folder, exist_ok=True)

    output_file = os.path.join(output_folder, 'output.json')
    output_data = {
        "conversation_history": conversation_history,
        "total_tokens": total_tokens_used,
        "image_indices": image_indices,
        "folder_name":folder_name
    }

    with open(output_file, 'w') as outfile:
        json.dump(output_data, outfile, indent=4)


process_json("ALFRED/1/conversation.json")
