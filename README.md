# document_summary
문서요약을 위한 챗봇입니다.
[KoAlpaca-Polyglot-5.8B](https://huggingface.co/beomi/KoAlpaca-Polyglot-5.8B) 모델을 [AIhub 문서 요약 데이터셋](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=97)에서 뉴스에 대한 데이터만을 fine-tuning 하였습니다.
# data preprocessing
```
import json
json_file_path = "Training/article_train_original.json"
save_path = "preprocessing/"
with open(json_file_path,'r') as json_file:
    json_dict = json.load(json_file)
    documents = json_dict['documents']


with open(f"{save_path}/article_train_docs.json",'w') as json_file:
    json.dump(documents, json_file)
```

# Requirement
* pytorch > 1.13.x
* transformers
* peft
* accelerate
* datasets

# Training
```
python main.py --dataset_path ./article_train_docs.json
```

# Web Demo
![image](https://github.com/mnbv7581/document_summary/assets/44501825/7b42502a-6e6f-4856-ba93-4cd6eaaf60e2)



