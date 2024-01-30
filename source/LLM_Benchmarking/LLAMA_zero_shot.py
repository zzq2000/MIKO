import json

import torch
import transformers
from tqdm import tqdm
from transformers import AutoTokenizer

with open('MIKO_benchmark.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

prompt_list = [
    "After posting this Tweet, the user wants to",
    "After viewing this Tweet, others will",
    "The user posts this Tweet because the user is",
    "The user posts this Tweet because the user intended to",
    "After posting this Tweet, the user feels",
    "After viewing this Tweet, others feel",
    "After viewing this Tweet, others want to",
    "After posting this Tweet, the user will",
    "Before posting this Tweet, the user needs to",
    "The user post this tweet because"
]

model_list = ["meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-2-13b-chat-hf", "meta-llama/Llama-2-7b-hf",
              "meta-llama/Llama-2-13b-hf", "tiiuae/falcon-7b-instruct",
              "mistralai/Mistral-7B-Instruct-v0.2", "mistralai/Mistral-7B-Instruct-v0.1", "mistralai/Mistral-7B-v0.1"]

for ind, model in enumerate(model_list):
    generations = []
    tokenizer = AutoTokenizer.from_pretrained(model, token="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")
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float32,
        device_map="auto",
        token="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",
    )

    for d in tqdm(data, desc="Generating for {} ({}/{})".format(model, ind + 1, len(model_list))):
        d['generations'] = {}
        for i in range(10):
            zero_shot_prompt = "Given a tweet sent by a user, generate the user's intention behind sending this tweet by completing the given sentence. Make the intention human-centric and focus on the mental intention.\n\nTweet: {}\n\nIntention: {}".format(
                d["text"], prompt_list[i])
            zero_shot_sequences = pipeline(
                zero_shot_prompt,
                do_sample=True,
                top_k=10,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                max_length=len(tokenizer.tokenize(zero_shot_prompt)) + 50,
                pad_token_id=tokenizer.pad_token_id)
            # print(zero_shot_sequences[0]['generated_text'])
            d['generations']['Intention {}'.format(i + 1)] = zero_shot_sequences[0]['generated_text']
        generations.append(d)

    with open('MIKO_benchmark_generation_{}.json'.format(model), 'w', encoding='utf-8') as f:
        json.dump(generations, f, indent=4, ensure_ascii=False)
