import openai
import os
import re

class Chat:
    def __init__(self) -> None:
        PROXY = '127.0.0.1:7890'
        os.environ['HTTP_PROXY'] = os.environ['http_proxy'] = PROXY
        os.environ['HTTPS_PROXY'] = os.environ['https_proxy'] = PROXY
        os.environ['NO_PROXY'] = os.environ['no_proxy'] = '127.0.0.1,localhost,.local'
        openai.api_key = "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"
        self.message = [
            {"role": "system", "content": "请你处理一个医疗领域命令实体识别的任务。可以抽取的实体类别包括疾病、临床表现、医疗程序、医疗设备、药物、医学检验项目、身体、科室、微生物类一共九类。对于接下来输入的所有文本，请你抽取出其中的命名实体，并输出其所属类别和在原文中的起始和终止位置。 (注:位置从0开始计数。)"},
        ]

    def get_response(self):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.message,
        )
        return response['choices'][0]['message']['content']

    def convert(self, text, result):
        result_list = result.split('\n')
        entities = []
        for entity in result_list[2:-1]:
            entity_name, entity_type, _, _ = entity.split('|')[1:-1]
            start_idxes = []
            end_idxes = []
            for pos in re.finditer(entity_name, text):
                start_idxes.append(pos.start())
                end_idxes.append(pos.end())
            for start_idx, end_idx in zip(start_idxes, end_idxes):
                entities.append({'entity':entity_name, 'type':entity_type, 'start_idx':start_idx, 'end_idx':end_idx})
        return entities

if __name__ == "__main__":
    PROXY = '127.0.0.1:7890'
    os.environ['HTTP_PROXY'] = os.environ['http_proxy'] = PROXY
    os.environ['HTTPS_PROXY'] = os.environ['https_proxy'] = PROXY
    os.environ['NO_PROXY'] = os.environ['no_proxy'] = '127.0.0.1,localhost,.local'
    openai.api_key = "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"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "请你处理一个医疗领域命令实体识别的任务。可以抽取的实体类别包括疾病、临床表现、医疗程序、医疗设备、药物、医学检验项目、身体、科室、微生物类一共九类。对于接下来输入的所有文本，请你抽取出其中的命名实体，并输出其所属类别和在原文中的起始和终止位置。 (注:位置从0开始计数。)"},
            {"role": "user", "content": "结缔组织疾病如风湿热患儿亦可发生。"},
            {"role": "assistant", "content": "请为句子中的每个位置预测一个实体类别"},
            {"role": "user", "content": "对儿童SARST细胞亚群的研究表明，与成人SARS相比，儿童细胞下降不明显，证明上述推测成立。",}
        ]
    )

    print(response['choices'][0]['message']['content'])