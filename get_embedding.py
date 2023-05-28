
from dataset import BartForScore, EEDataset, CollateFnForEE
from transformers import BertTokenizer
import torch
from tqdm import tqdm
from kmeans_pytorch import kmeans
import operator
from functools import reduce
import json

def get_bart_model(root):
    model = BartForScore.from_pretrained(root)
    tokenizer = BertTokenizer.from_pretrained(root)
    return model, tokenizer

def embeded(model, dataset, loader, device, mean=None, num_train=0):
    if mean is None:
        features = []
        model.eval()
        with torch.no_grad():
            for data in tqdm(dataset):
                data = loader([data])
                input_ids, attention_mask = data['input_ids'].to(device), data['attention_mask'].to(device)
                output = model(input_ids=input_ids, attention_mask=attention_mask)
                features.append(output)
        features = torch.stack(features)
        return features
    else:
        dist = []
        model.eval()
        with torch.no_grad():
            for data in tqdm(dataset):
                data = loader([data])
                input_ids, attention_mask = data['input_ids'].to(device), data['attention_mask'].to(device)
                output = model(input_ids=input_ids, attention_mask=attention_mask)
                dist.append(torch.nn.functional.cosine_similarity(mean, output))
        dist = torch.stack(dist).mean(dim=-1)
        a, idx_ = torch.sort(dist, descending=True)
        return idx_[:num_train].cpu().tolist()



def cluster(num_cluster, features, device):
    cluster_ids_x, cluster_centers = kmeans(X=features, num_clusters=num_cluster, device=device)
    return cluster_centers.to(device)


if __name__ == "__main__":
    num_cluster = 50 # dev集合聚类个数
    num_train = 60 # 我们提供的prompt训练样本的数量
    model_root = '/home/luhaotian/AI3612/cmeee/bart-base-chinese'
    data_root = '/home/luhaotian/AI3612/cmeee/data/CBLUEDatasets'
    device = 'cuda:0'

    model, tokenizer = get_bart_model(model_root)
    model = model.to(device)
    train_set = EEDataset(data_root, 'train', tokenizer)
    dev_set = EEDataset(data_root, 'select_dev', tokenizer)
    loader = CollateFnForEE(tokenizer.pad_token_id)

    dev_features = embeded(model, dev_set, loader, device, mean=None, num_train=num_train)
    means = cluster(num_cluster, dev_features, device)
    train_idx = embeded(model, train_set, loader, device, mean=means, num_train=num_train)

    use_set = []
    for idx_ in train_idx:
        use_set.append(train_set.idx[idx_])
    print(len(use_set))
    with open('./select_train.json', 'w', encoding="utf8") as f:
        json.dump(use_set, f, ensure_ascii=False, indent=4)


