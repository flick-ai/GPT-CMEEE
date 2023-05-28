from dataset import EEDataloader
import random
from tqdm import tqdm
from get_response import Chat
import time
from collections import Counter
import json

def ensemle(result):
    # input : list : 10 * entities
    # entities : a list with some dicts
    # dict: start_idx end_idx type entity
    # output : an entity [some dicts]
    # 将10个结果进行投票，获取最终结果
    # 不考虑相同坐标识别type和entity不同的情况，如果出现就随机选择
    output = []

    num_entities = len(result) # 10
    # 首先按照start_idx和end_idx对进行投票，然后筛选出出现次数大于5的实体
    idxs = []
    for netities in result:
        for entity in netities:
            idxs.append((entity['start_idx'], entity['end_idx']))
    idxs_counter = Counter(idxs)
    idxs = []
    #筛选出出现次数大于一半的实体
    for idx in idxs_counter:
        if idxs_counter[idx] >= num_entities / 2:
            idxs.append({'start_idx':idx[0], 'end_idx':idx[1]})

    # 然后按照type和entity进行投票
    for idx in idxs:
        tys = []
        entitys = []
        for netities in result:
            for entity in netities:
                if entity['start_idx'] == idx['start_idx'] and entity['end_idx'] == idx['end_idx']:
                    tys.append(entity['type'])
                    entitys.append(entity['entity'])
        ty = Counter(tys).most_common(1)[0][0]
        entity = Counter(entitys).most_common(1)[0][0]
        output.append({"start_idx":idx['start_idx'], "end_idx":idx['end_idx'], "type":ty, "entity": entity})

    return output

if __name__ == "__main__":

    # 读取测试样本
    dev_examples = EEDataloader().get_data(path="./select_dev.json")
    dev_dataset = []
    for example in dev_examples[0:100]:
        text, prompt = example.to_gpt_task()
        dev_dataset.append({"text":text, "prompt":prompt, 'target':example.entities})

    # 读取选择的训练样本
    dataset_path = "./select_train.json"
    train_examples = EEDataloader().get_data(path=dataset_path)
    train_dataset = []
    for example in train_examples:
        text, prompt = example.to_gpt_task()
        train_dataset.append({"text":text, "prompt":prompt})

    response_time = 0
    # 遍历测试样本：
    for idx, dev_data in tqdm(enumerate(dev_dataset), total=len(dev_dataset)):
        # 随机采样10个训练样本
        train_data = random.choices(train_dataset, k=10)
        correct_data = random.choices(train_dataset, k=1)[0]
        # 进行10次重新排序，喂给chatgpt进行响应
        results = []
        for i in range(3):
            # 实例化一个对话对象
            chatgpt = Chat()
            # 将训练数据的prompt喂给chatgpt
            random.shuffle(train_data)
            for data in train_data:
                chatgpt.message.append({"role":"user", "content":data['text']})
                chatgpt.message.append({"role":"assistant", "content":data['prompt']})
            # # 将纠错数据的prompt喂给chatgpt
            # chatgpt.message.append({"role":"user", "content":correct_data['text']})
            # response = chatgpt.get_response()
            # chatgpt.message.append({"role":"user", "content":"正确的实体识别:"+correct_data['prompt']})
            # 将测试数据的text喂给chatgpt
            chatgpt.message.append({"role":"user", "content":dev_data['text']})
            response_time += 1
            if response_time % 3 == 0:
                time.sleep(80)
            result = chatgpt.get_response()
            # 将相应数据转换为标准格式
            results.append(chatgpt.convert(dev_data['text'], result))
        ensemble_output = ensemle(results)
        dev_dataset[idx]['output'] = ensemble_output

    with open('./output.json', 'w', encoding="utf8") as f:
        json.dump(dev_dataset, f, ensure_ascii=False, indent=4)

