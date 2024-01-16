import json
with open('train.json', 'r', encoding='utf-8') as f:
    train = json.load(f)
    print(len(train))
with open('valid.json', 'r', encoding='utf-8') as f:
    valid = json.load(f)
    print(len(valid))
with open('test.json', 'r', encoding='utf-8') as f:
    test = json.load(f)
    print(len(test))
def load_jsonl(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data 
# imageid2label = {}
# for i in range(len(train)):
#     imageid2label[train[i]['image']] = train[i]['label']
# for i in range(len(valid)):
#     imageid2label[valid[i]['image']] = valid[i]['label']
# for i in range(len(test)):
#     imageid2label[test[i]['image']] = test[i]['label']
# train_intention = load_jsonl('Generation/keyinfo_train1_intention19.jsonl')
# valid_intention = load_jsonl('Generation/keyinfo_valid1_intention20.jsonl')
# test_intention = load_jsonl('Generation/keyinfo_test1_intention20.jsonl')
# # add label for train_intention using train
# for i in range(len(train_intention)):
#     train_intention[i]['label'] = imageid2label[train_intention[i]['image_id']]
# # add label for valid_intention using valid
# for i in range(len(valid_intention)):
#     valid_intention[i]['label'] = imageid2label[valid_intention[i]['image_id']]
# # add label for test_intention using test
# for i in range(len(test_intention)):
#     test_intention[i]['label'] = imageid2label[test_intention[i]['image_id']]
# # save train_intention
# with open('train_intention.jsonl', 'w', encoding='utf-8') as f:
#     for line in train_intention:
#         json.dump(line, f)
#         f.write('\n')
# # save valid_intention
# with open('valid_intention.jsonl', 'w', encoding='utf-8') as f:
#     for line in valid_intention:
#         json.dump(line, f)
#         f.write('\n')
# # save test_intention
# with open('test_intention.jsonl', 'w', encoding='utf-8') as f:
#     for line in test_intention:
#         json.dump(line, f)
#         f.write('\n')

    
    