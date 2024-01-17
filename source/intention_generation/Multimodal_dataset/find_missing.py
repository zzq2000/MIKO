import json

name_list = ['test', 'train', 'valid', 'lost']
for ind, (data1, data2) in enumerate([('./keyinfo_test1.jsonl', './keyinfo_test1_intention20.jsonl'),
                                      ('./keyinfo_train1.jsonl', './keyinfo_train1_intention19.jsonl'),
                                      ('./keyinfo_valid1.jsonl', './keyinfo_valid1_intention20.jsonl'),
                                      ('./lost_data_in_train.jsonl', './lost_data_intention19.jsonl')]):
    # read two jsonl files
    data1_json = [json.loads(line) for line in open(data1, 'r')]
    data2_json = [json.loads(line) for line in open(data2, 'r')]
    # find the missing data
    data1_id = [data['question_id'] for data in data1_json]
    data2_id = [data['question_id'] for data in data2_json]

    missing_id = []
    for id in data1_id:
        if id not in data2_id:
            missing_id.append(id)
    print(len(missing_id))

    # write the missing data into a jsonl file
    missing_data = []
    for data in data1_json:
        if data['question_id'] in missing_id:
            missing_data.append(data)
    print(len(missing_data))

    with open('./missing_data_{}.jsonl'.format(name_list[ind]), 'w') as f:
        for data in missing_data:
            json.dump(data, f)
            f.write('\n')
