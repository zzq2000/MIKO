import json

name_list = ['valid', 'test', 'train']
for ind, (data1, data2) in enumerate(
        [('./missing_data_valid_intention_make_up.jsonl', './keyinfo_valid1_intention20.jsonl'),
         ('./missing_data_test_intention_make_up.jsonl', './keyinfo_test1_intention20.jsonl'),
         ('./missing_data_train_intention_make_up.jsonl', './keyinfo_train1_intention19.jsonl')]):
    # read two jsonl files
    data1_json = [json.loads(line) for line in open(data1, 'r')]
    data2_json = [json.loads(line) for line in open(data2, 'r')]

    total_data = data1_json + data2_json
    if name_list[ind] == 'train':
        data3_json = [json.loads(line) for line in open('./missing_data_lost_intention_make_up.jsonl', 'r')]
        data4_json = [json.loads(line) for line in open('./lost_data_intention19.jsonl', 'r')]
        total_data = total_data + data3_json + data4_json
        # print(len(data1_json), len(data2_json), len(data3_json), len(data4_json))

    # sort total_data according to item['question_id'] of each item in total_data
    total_data = sorted(total_data, key=lambda x: x['question_id'])
    print("split:{}".format(name_list[ind]), len(total_data))

    with open('./final_generation/merge_data_{}.jsonl'.format(name_list[ind]), 'w') as f:
        for data in total_data:
            json.dump(data, f)
            f.write('\n')
