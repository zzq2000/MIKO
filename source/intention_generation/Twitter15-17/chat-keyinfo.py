import openai
import json
import re
import time
openai.api_base = 'https://api.closeai-proxy.xyz/v1'
openai.api_key = 'sk-Y1Makaf6u43RXnCujsaKlOl7jCfu4W8I3kpOfaKIi6Z6qGDY'

def extract_information(text):
    # 初始化结果字典
    result = {
        "concept": None,
        "action": None,
        "object": None,
        "emotion": None,
        "keywords": None
    }
    # 在文本中查找并提取相关信息
    lines = text.split("\n")
    for line in lines:
        if line.startswith("Concept:"):
            result["concept"] = line.replace("Concept:", "").strip()
        elif line.startswith("Action:"):
            result["action"] = line.replace("Action:", "").strip()
        elif line.startswith("Object:"):
            result["object"] = line.replace("Object:", "").strip()
        elif line.startswith("Emotion:"):
            result["emotion"] = line.replace("Emotion:", "").strip()
        elif line.startswith("Keywords:"):
            result["keywords"] = line.replace("Keywords:", "").strip()
    return result

def format_text(text,image_des,key_list):
    #可以对prompt进行修改，引导模型生成更好的intention，目前每个样本生成5个intention
    formatted_text = "Based on the below information, please describe the intention of why user to posted this information.Generate all intents possible.The information are as follows:\n"
    formatted_text += "Text: " + text + "\n"
    formatted_text += "Image description: " + image_des + "\n"
    formatted_text += "Concept: " + key_list["concept"] + ".\n"
    formatted_text += "Action: " + key_list["action"] + ".\n"
    formatted_text += "Object: " + key_list["object"] + ".\n"
    formatted_text += "Emotion: " + key_list["emotion"] + ".\n"
    formatted_text += "Keywords: " + key_list["keywords"] + ".\n"
    formatted_text += "Note: please describe it as brief as possible and make it universal."
    # formatted_text += "Note: Please generate intention in JSON format!"
    return formatted_text

def format_keyinfo(key_list):
    #把关键信息保存下来
    formatted_text1 = ""
    formatted_text1 += "Concept: " + key_list["concept"] + ".\n"
    formatted_text1 += "Action: " + key_list["action"] + ".\n"
    formatted_text1 += "Object: " + key_list["object"] + ".\n"
    formatted_text1 += "Emotion: " + key_list["emotion"] + ".\n"
    formatted_text1 += "Keywords: " + key_list["keywords"] + ".\n"
    return formatted_text1

def read_multi_line_json(file_path):
    with open(file_path, 'r') as file:
        json_data = []
        for line in file:
            json_data.append(json.loads(line))
    return json_data

# 指定多行JSON文件的输入路径，可以根据文件的位置进行修改
json_file_path = 'F:/Intention/llava_stage_two_keyinfo-prompt/eg.json'

# 读取多行JSON文件
data = read_multi_line_json(json_file_path)
# print(data)
# 打印读取的JSON数据
result = []
for item in data:
    time.sleep(1)
    text = item["prompt"]
    print(item["question_id"])
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "system",
         "content": "You are a chatbot,you should try your best to help users understand the purpose of posting"},
        {"role": "user",
         "content": text}
      ]
    )
    response_out = response["choices"][-1]["message"]["content"]
    # print(response_out)
    zz = extract_information(response_out)
    key_info_prompt_intention = format_text(item["text"],item["image_descrption"],zz)
    key_info_save = format_keyinfo(zz)
    json_data = {
        "question_id": item["question_id"],
        "text": item["text"],
        "image_descrption":item["image_descrption"],
        "prompt": key_info_prompt_intention}
    # print(json)
    response1 = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": "You are a chatbot,you should try your best to help users understand the purpose of posting"},
            {"role": "user",
             "content": json_data["prompt"]}
        ],
        temperature=0.5
    )
    response_out1 = response1["choices"][-1]["message"]["content"]
    json_data = {
        "question_id": item["question_id"],
        "image_id": item["image_id"],
        "text": item["text"],
        "image_descrption": item["image_descrption"],
        "prompt": key_info_prompt_intention,
        "keyinfo": key_info_save,
        "intention": response_out1}
    result.append(json_data)
    #最终输出的路径，进行了二三步的处理，既保存了关键信息，有根据关键信息，把intention保存下来了（目前每个样本保存5个不同的intention）
file_name = 'F:/Intention/llava_stage_two_keyinfo-prompt/eg8_out.json'
with open(file_name, 'w') as json_file:
    for item in result:
        print(item["question_id"])
        json.dump(item, json_file)
        json_file.write('\n')  # 在对象之间添加换行符
print(f"数据已保存到 {file_name}")

