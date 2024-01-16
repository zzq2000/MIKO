import argparse
import os
import json
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["WANDB_PROJECT"]="LLM4SCT"

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_recall_curve,f1_score,recall_score,precision_score
import numpy as np
import pandas as pd


def seed_everything(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)

def metrics(eval_pred):
    logits, labels = eval_pred
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
    acc = accuracy_score(labels, np.argmax(probs, axis=1))
    probs = probs[:, 1]
    auc = roc_auc_score(labels, probs, multi_class='ovr')
    pr_auc = average_precision_score(labels, probs)
    f1 = f1_score(labels, np.argmax(probs, axis=1))
    recall = recall_score(labels, np.argmax(probs, axis=1))
    precision = precision_score(labels, np.argmax(probs, axis=1))
    return {"roc_auc": auc, "pr_auc": pr_auc, 'acc': acc, 'f1': f1, 'recall': recall, 'precision': precision}

def load_jsonl(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", '--out', default='ours')
    parser.add_argument("--model", default='bert-base-cased')
    # parser.add_argument("--size", default='base')
    parser.add_argument("--bs", default=16, type=int)
    parser.add_argument("--metric", default='loss')
    parser.add_argument("--epoch", default=10, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--grad_accu_steps", type=int, default=1)
    parser.add_argument("--type", type=int, default=1)
    args = parser.parse_args()
    seed_everything()

    model_checkpoint = args.model
    train = load_jsonl('train_intention.jsonl')
    ds_train = pd.DataFrame(train)
    
    valid = load_jsonl('valid_intention.jsonl')
    ds_valid = pd.DataFrame(valid)

    test = load_jsonl('test_intention.jsonl')
    ds_test = pd.DataFrame(test)
    # all_df['text'] = all_df['text'].str.replace('\n', '')
    # delete empty text
    # all_df = all_df[all_df['text'] != '']
    # split train and test randomly 8:2
    # train = all_df.sample(frac=frac, random_state=seed)
    # test = all_df.drop(train.index)
    # train, test = get_data('data_html.json')
    
    print(ds_train)
    print(ds_valid)
    print(ds_test)
    # merge text image_descrption intention to text
    if args.type == 1:
        print('merge text image_descrption intention to text')
        ds_train['text'] = ds_train['text'] + ds_train['image_descrption'] + ds_train['intention']
        ds_valid['text'] = ds_valid['text'] + ds_valid['image_descrption'] + ds_valid['intention']
        ds_test['text'] = ds_test['text'] + ds_test['image_descrption'] + ds_test['intention']
    elif args.type == 2:
        print('merge text image_descrption to text')
        ds_train['text'] = ds_train['text'] + ds_train['image_descrption']
        ds_valid['text'] = ds_valid['text'] + ds_valid['image_descrption']
        ds_test['text'] = ds_test['text'] + ds_test['image_descrption']
    elif args.type == 3:
        print('merge text intention to text')
        ds_train['text'] = ds_train['text'] + ds_train['intention']
        ds_valid['text'] = ds_valid['text'] + ds_valid['intention']
        ds_test['text'] = ds_test['text'] + ds_test['intention']
    elif args.type == 4:
        print('only text')
        ds_train['text'] = ds_train['text']
        ds_valid['text'] = ds_valid['text']
        ds_test['text'] = ds_test['text']
    elif args.type == 5:
        print('only image_descrption')
        ds_train['text'] = ds_train['image_descrption']
        ds_valid['text'] = ds_valid['image_descrption']
        ds_test['text'] = ds_test['image_descrption']
    elif args.type == 6:
        print('only intention')
        ds_train['text'] = ds_train['intention']
        ds_valid['text'] = ds_valid['intention']
        ds_test['text'] = ds_test['intention']
    elif args.type == 7:
        print('merge image_descrption intention to text')
        ds_train['text'] =  ds_train['image_descrption'] + ds_train['intention']
        ds_valid['text'] =  ds_valid['image_descrption'] + ds_valid['intention']
        ds_test['text'] =  ds_test['image_descrption'] + ds_test['intention']

    # remove image_description and intention
    # ds_train = ds_train.remove_columns(['image_descrption', 'intention'])
    # ds_valid = ds_valid.remove_columns(['image_descrption', 'intention'])
    # ds_test = ds_test.remove_columns(['image_descrption', 'intention'])
    print(ds_train)
    print(ds_valid)
    print(ds_test)
    ds_train = Dataset.from_pandas(ds_train)
    ds_valid = Dataset.from_pandas(ds_valid)
    ds_test = Dataset.from_pandas(ds_test)
    

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)


    def preprocess_function(examples):
        return tokenizer(examples['text'], max_length=512, padding=True, truncation=True)


    ds_train_enc = ds_train.map(preprocess_function, batched=True)
    ds_valid_enc = ds_valid.map(preprocess_function, batched=True)
    ds_test_enc = ds_test.map(preprocess_function, batched=True)

    num_labels = 2
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
    metric_name = args.metric
    model_name = args.model
    batch_size = args.bs
    strategy = 'steps' if args.eval_steps > 0 else 'epoch'
    train_args = TrainingArguments(
        os.path.join('output', f"{model_name}-finetuned-{args.out}"),
        evaluation_strategy=strategy,
        save_strategy=strategy,
        learning_rate=args.lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=args.grad_accu_steps,  # 用了这个会每8个step进行一次反向传播
        num_train_epochs=args.epoch,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        report_to="wandb",  # enable logging to W&B
        run_name=args.model,  # name of the W&B run (optional)
        logging_steps=1,  # how often to log to W&B
    )

    trainer = Trainer(
        model,
        train_args,
        train_dataset=ds_train_enc,
        eval_dataset=ds_valid_enc,
        tokenizer=tokenizer,
        compute_metrics=metrics
    )
    trainer.train()
    trainer.evaluate(test_dataset=ds_test_enc)
    trainer.save_model(os.path.join('saved_models', args.out))
    model.save_pretrained(os.path.join('saved_models', args.out + "_test"))
