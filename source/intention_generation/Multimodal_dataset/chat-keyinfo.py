import glob
import json
import sys
import time

import numpy as np
import openai
from tqdm import tqdm

sys.path.append('../')

from key import *

openai.api_key = weiqi_primary_key
openai.api_base = "https://hkust.azure-api.net"
openai.api_type = "azure"
openai.api_version = "2023-05-15"
model = "gpt-35-turbo"


def generate_with_openai(prompt, system_prompt=None, max_tokens=100, temperature=0.5, top_p=1.0, retry_attempt=4,
                         verbose=False):
    if system_prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
    else:
        messages = [
            {"role": "user", "content": prompt},
        ]
    retry_num = 0
    generation_success = False
    while retry_num < retry_attempt and not generation_success:
        try:
            gen = openai.ChatCompletion.create(
                engine=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            generation_success = True
            total_tokens = gen['usage']['total_tokens']
            if verbose:
                # print total tokens
                print("Total tokens: {}".format(total_tokens))
        except openai.error.APIError as e:
            if verbose:
                print(e)
            retry_num += 1
            generation_success = False
            time.sleep(1)
        except openai.error.InvalidRequestError as e:
            if verbose:
                print(e)
            retry_num += 1
            generation_success = False
            time.sleep(1)
        except openai.error.APIConnectionError as e:
            if verbose:
                print(e)
            retry_num += 1
            generation_success = False
            time.sleep(1)
        except openai.error.RateLimitError as e:
            if verbose:
                print(e)
            retry_num += 1
            generation_success = False
            time.sleep(30)
    if generation_success:
        try:
            return True, (gen['choices'][0]['message']['content'].strip(), total_tokens)
        except:
            return False, None
    else:
        return False, None


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


def format_text(text, image_des, key_list):
    # 可以对prompt进行修改，引导模型生成更好的intention，目前每个样本生成5个intention
    formatted_text = "Based on the information below, guess the intention of why the user post this information. Generate one different intentions if possible. The information is:\n"
    formatted_text += "Text: " + text + "\n"
    formatted_text += "Image description: " + image_des + "\n"
    try:
        formatted_text += "Concept: " + key_list["concept"] + ".\n"
    except:
        pass
    try:
        formatted_text += "Action: " + key_list["action"] + ".\n"
    except:
        pass
    try:
        formatted_text += "Object: " + key_list["object"] + ".\n"
    except:
        pass
    try:
        formatted_text += "Emotion: " + key_list["emotion"] + ".\n"
    except:
        pass
    try:
        formatted_text += "Keywords: " + key_list["keywords"] + ".\n"
    except:
        pass
    # if relation == 'xWant':
    #     formatted_text += "You can think about the concepts, actions, object, emotions, and keywords. Make the intention human-centric, and formulate your answer as: After posting this Tweet, the user wants to ..."
    # elif relation == 'oEffect':
    #     formatted_text += "You can think about the concepts, actions, object, emotions, and keywords. Make the intention human-centric, and formulate your answer as: After viewing this Tweet, others will ..."
    # elif relation == 'xAttr':
    #     # formatted_text += "You can think about the concepts, actions, object, emotions, and keywords. Make the intention human-centric, and formulate your answer as: The user posts this Tweet because the user is ..."
    #     formatted_text += "You can think about the concepts, actions, object, emotions, and keywords. Make the intention human-centric, and formulate your answer as: Intention: ..."
    # elif relation == 'xIntent':
    #     formatted_text += "You can think about the concepts, actions, object, emotions, and keywords. Make the intention human-centric, and formulate your answer as: The user posts this Tweet because the user intended to ..."
    # elif relation == 'xReact':
    #     formatted_text += "You can think about the concepts, actions, object, emotions, and keywords. Make the intention human-centric, and formulate your answer as: After posting this Tweet, the user feels ..."
    # elif relation == 'oReact':
    #     formatted_text += "You can think about the concepts, actions, object, emotions, and keywords. Make the intention human-centric, and formulate your answer as: After viewing this Tweet, others feel ..."
    # elif relation == 'oWant':
    #     formatted_text += "You can think about the concepts, actions, object, emotions, and keywords. Make the intention human-centric, and formulate your answer as: After viewing this Tweet, others want to ..."
    # elif relation == 'xEffect':
    #     formatted_text += "You can think about the concepts, actions, object, emotions, and keywords. Make the intention human-centric, and formulate your answer as: After posting this Tweet, the user will ..."
    # elif relation == 'xNeed':
    #     formatted_text += "You can think about the concepts, actions, object, emotions, and keywords. Make the intention human-centric, and formulate your answer as: Before posting this Tweet, the user needed ..."
    formatted_text += (
        "You can think about the concepts, actions, object, emotions, and keywords. Make the intention human-centric, and generate ten intentions by formulating your answer as:\n"
        "Intention1: After posting this Tweet, the user wants to ...\n"
        "Intention2: After viewing this Tweet, others will ...\n"
        "Intention3: The user posts this Tweet because the user is ...\n"
        "Intention4: The user posts this Tweet because the user intended to ...\n"
        "Intention5: After posting this Tweet, the user feels ...\n"
        "Intention6: After viewing this Tweet, others feel ...\n"
        "Intention7: After viewing this Tweet, others want to ...\n"
        "Intention8: After posting this Tweet, the user will ...\n"
        "Intention9: Before posting this Tweet, the user needs to ...\n"
        "Intention10: The user post this tweet because...\n")
    return formatted_text


def format_keyinfo(key_list):
    # 把关键信息保存下来
    formatted_text1 = ""
    try:
        formatted_text1 += "Concept: " + key_list["concept"] + ".\n"
    except:
        pass
    try:
        formatted_text1 += "Action: " + key_list["action"] + ".\n"
    except:
        pass
    try:
        formatted_text1 += "Object: " + key_list["object"] + ".\n"
    except:
        pass
    try:
        formatted_text1 += "Emotion: " + key_list["emotion"] + ".\n"
    except:
        pass
    try:
        formatted_text1 += "Keywords: " + key_list["keywords"] + ".\n"
    except:
        pass
    return formatted_text1


def read_multi_line_json(file_path):
    with open(file_path, 'r') as file:
        json_data = []
        for line in file:
            json_data.append(json.loads(line))
    return json_data


json_files = glob.glob('./*.jsonl')

TEST_MODE = False
VERBOSE = False
RELATION_LIST = ['xWant', 'oEffect', 'xAttr', 'xIntent', 'xReact', 'oReact', 'oWant', 'xEffect', 'xNeed']
NUM_OF_TEST = 3  # number of example to test
# 指定多行JSON文件的输入路径，可以根据文件的位置进行修改
for json_file_path in json_files:
    if 'intention' in json_file_path:
        continue
    failed_generation_id = []
    total_tokens_file = 0
    # 读取多行JSON文件
    data = read_multi_line_json(json_file_path)
    if TEST_MODE:
        data = data[:NUM_OF_TEST]
    # print(data)
    # 打印读取的JSON数据
    result = []
    for item in tqdm(data, desc=json_file_path):
        time.sleep(1)
        text = item["prompt"]
        # print(item["question_id"])
        # print(text)

        generation_status, keyword_generation = generate_with_openai(
            text + " Formulate your answer as: Concept:\nAction:\nObject:\nEmotion:\nKeywords:\n", max_tokens=300,
            retry_attempt=3,
            verbose=VERBOSE)
        if not generation_status:
            failed_generation_id.append(item["question_id"])
            continue
        # print(response_out)
        zz = extract_information(keyword_generation[0])
        total_tokens_file += keyword_generation[1]
        key_info_prompt_intention = format_text(item["text"], item["image_descrption"], zz)
        key_info_save = format_keyinfo(zz)
        json_data = {
            "question_id": item["question_id"],
            "text": item["text"],
            "image_descrption": item["image_descrption"],
            "prompt": key_info_prompt_intention}
        # print(json)
        generation_status_intention, intention_generation = generate_with_openai(json_data["prompt"], max_tokens=300,
                                                                                 temperature=0.7, retry_attempt=3,
                                                                                 verbose=VERBOSE)
        if not generation_status_intention:
            # append the failed generation id and corresponding relation
            failed_generation_id.append(item["question_id"])
            continue
        total_tokens_file += intention_generation[1]
        if TEST_MODE:
            print(intention_generation[0])
        json_data = {
            "question_id": item["question_id"],
            "image_id": item["image_id"],
            "text": item["text"],
            "image_descrption": item["image_descrption"],
            "prompt": key_info_prompt_intention,
            "keyinfo": key_info_save,
            "intention": intention_generation[0]}
        result.append(json_data)
    # for relation in RELATION_LIST:
    #     key_info_prompt_intention = format_text(item["text"], item["image_descrption"], zz, relation)
    #     key_info_save = format_keyinfo(zz)
    #     json_data = {
    #         "question_id": item["question_id"],
    #         "text": item["text"],
    #         "image_descrption": item["image_descrption"],
    #         "prompt": key_info_prompt_intention}
    #     # print(json)
    #     generation_status_intention, intention_generation = generate_with_openai(json_data["prompt"], max_tokens=300,
    #                                                                             temperature=0.7, retry_attempt=3,
    #                                                                             verbose=VERBOSE)
    #     if not generation_status_intention:
    #         # append the failed generation id and corresponding relation
    #         failed_generation_id.append(item["question_id"]+"_"+relation)
    #         continue
    #     total_tokens_file += intention_generation[1]
    #     if TEST_MODE:
    #         print(intention_generation[0])
    #     json_data = {
    #         "question_id": item["question_id"],
    #         "image_id": item["image_id"],
    #         "text": item["text"],
    #         "image_descrption": item["image_descrption"],
    #         "prompt": key_info_prompt_intention,
    #         "keyinfo": key_info_save,
    #         "intention": intention_generation[0],
    #         "relation": relation}  # TODO: add relation here for future parsing
    #     result.append(json_data)
    # 最终输出的路径，进行了二三步的处理，既保存了关键信息，有根据关键信息，把intention保存下来了（目前每个样本保存5个不同的intention）

    # for every five percent of number of data, save the result
    # if len(result) % (len(data) // 20) == 0:
    #     file_name = json_file_path.replace(".json", "_intention.json")
    #     with open(file_name, 'w') as json_file:
    #         for item in result:
    #             print(item["question_id"])
    #             json.dump(item, json_file)
    #             json_file.write('\n')  # 在对象之间添加换行符
    #     print(f"Generations saved into {file_name}")
    #     print("For file {}, total tokens: {}".format(json_file_path, total_tokens_file))
    #     np.save(json_file_path.replace(".json", "_failed_generation_id.npy"), failed_generation_id)
    if TEST_MODE:
        file_name = json_file_path.replace(".json", "_test_intention.json")
        with open(file_name, 'w') as json_file:
            for item in result:
                print(item["question_id"])
                json.dump(item, json_file)
                json_file.write('\n')  # 在对象之间添加换行符
        print(f"Generations saved into {file_name}")
        print("For file {}, total tokens: {}".format(json_file_path, total_tokens_file))
        # write them into a log txt
        with open(json_file_path.replace(".jsonl", "_intention_generation_log.txt"), 'w') as log_file:
            log_file.write("Total tokens: {}".format(total_tokens_file))
            log_file.write("\n")
            log_file.write("Failed generation id: {}".format(failed_generation_id))
        np.save(json_file_path.replace(".json", "_test_failed_generation_id.npy"), failed_generation_id)
