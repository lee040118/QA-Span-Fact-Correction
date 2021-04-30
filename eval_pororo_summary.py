import json, re
import pandas as pd
from pororo import Pororo
from evaluation import FactCorrect
from transformers import (
    ElectraTokenizer,
)
from rouge import Rouge
def load_json(file_path):
    return json.load(open(file_path, 'r'))

def save_data(data):
    output_file = "./result_cate_pororo.jsonl"
    with open(output_file, "w", encoding="utf-8") as fd:
        for example in (data):
            example = dict(example)
            fd.write(json.dumps(example, ensure_ascii=False) + "\n")
def normalize_answer(s):
    s = re.sub("“", "", s)
    s = re.sub("\"", "", s)
    s = re.sub("”", "", s)
    s = " ".join(re.split(r"\s+", s))
#     s = re.sub(" ”","”", s)
    return s

model_name_or_path = 'ckpt/koelectra-small-v3-korquad-ckpt-ver5/checkpoint-35700'
model = FactCorrect(model_name_or_path)
summ = Pororo(task='summarization', model="abstractive", lang="ko")
ner = Pororo(task="ner", lang="ko")
tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

filepath = './test.json'
docs = load_json(filepath)
docs = pd.DataFrame(docs, columns=["id", "text", "summary"])

new_data = []
long = 0
cnt, good_cnt, bad_cnt, no_cnt, wrong_cnt, v_cnt = 0, 0, 0, 0, 0, 0
r = Rouge()
scores = []
before_scores = []
# print(ner_q)
for i,idx in enumerate(range(len(docs),0,-1)):
    print(i)
    instance = docs.iloc[idx - 1]
    doc = instance['text']
    doc = normalize_answer(doc)
    corrupt = summ(doc)
    corrupt = normalize_answer(corrupt)
    tok_len = len(tokenizer.tokenize(corrupt))
    if tok_len > 64:
        long += 1
        continue

    before_scores += r.get_scores(corrupt, instance['summary'])

    ner_sum = ner(corrupt)
    categories = ("PERSON", "LOCATION", "ORGANIZATION", "CIVILIZATION", "DATE", "TIME", "QUANTITY")
    swap_sum = corrupt
    if corrupt == instance['summary']:
        label = "CORRECT"
    else:
        label ="INCORRECT"
    example = dict(instance)
    sp = 0
    for tar, ent in ner_sum:
        if ent != 'O' and ent in categories:
            ep = sp+len(tar)
            print("masked ent : {}".format(tar))
            mask_s = swap_sum[:sp] + '[MASK]' + swap_sum[ep:]
            print("masking : {}".format(mask_s))
            ans = model.predict(doc, mask_s)
            swap_sum = mask_s[:sp] + ans[0] + mask_s[sp+6:]
            sp += len(ans[0])
            print("predict : {}".format(ans[0]))
            print("swaping : {}".format(swap_sum))
            print()
        else:
            sp += len(tar)

    scores += r.get_scores(swap_sum, instance['summary'])
    example["qa_correct"] = swap_sum
    print("정답 : {}".format(instance["summary"]))
    print("잘못 : {}".format(corrupt))
    print("swap : {}".format(swap_sum))
    # print()
    if label != "CORRECT" and swap_sum == instance["summary"]:
        # print("good")
        cnt +=1
        good_cnt+=1
    elif label != "CORRECT" and corrupt != swap_sum and swap_sum != instance["summary"]:
        cnt+=1
        bad_cnt+=1
    elif corrupt == swap_sum and  label != "CORRECT":
        no_cnt+=1
    elif label == "CORRECT" and swap_sum != corrupt and instance["summary"] != swap_sum:
        cnt+=1
        wrong_cnt+=1
    elif label == "CORRECT" and swap_sum == corrupt:
        v_cnt+=1

    new_data.append(example)

rouge_1_f, rouge_2_f, rouge_l_f = 0, 0, 0

for num in before_scores:
    rouge_1_f += num['rouge-1']['f']
    rouge_2_f += num['rouge-2']['f']
    rouge_l_f += num['rouge-l']['f']
rouge_1_f /= len(before_scores)
rouge_2_f /= len(before_scores)
rouge_l_f /= len(before_scores)

print(f'before rouge_1_f : {rouge_1_f} rouge_2_f : {rouge_2_f} rouge-l : {rouge_l_f}')
new_data.append({"before rouge_1_f" : rouge_1_f, "before rouge_2_f": rouge_2_f, "before rouge_l_f": rouge_l_f})

for num in scores:
    rouge_1_f += num['rouge-1']['f']
    rouge_2_f += num['rouge-2']['f']
    rouge_l_f += num['rouge-l']['f']

rouge_1_f /= len(scores)
rouge_2_f /= len(scores)
rouge_l_f /= len(scores)

print(f'rouge_1_f : {rouge_1_f} rouge_2_f : {rouge_2_f} rouge-l : {rouge_l_f}')
new_data.append({"rouge_1_f": rouge_1_f, "rouge_2_f": rouge_2_f, "rouge_l_f": rouge_l_f})

print(f' 모델이 수정할 확률 : {cnt / (len(docs)- long)}')
print(f' 잘못 생성된 요약문을 모델이 잘 수정함 {good_cnt / (len(docs)- long)}')
print(f' 잘못 생성된 요약문을 모델이 잘 못 수정함 {bad_cnt / (len(docs)- long)}')
print(f' 잘못 생성된 요약문을 모델이 수정 안함 {no_cnt / (len(docs)- long)}')
print(f' 잘 생성된 요약문을 모델이 잘못 수정함 {wrong_cnt / (len(docs)- long)}')
print(f' 잘 생성된 요약문을 모델이 수정안함 {v_cnt / (len(docs)- long)}')
print(f' 정확도 {(good_cnt + v_cnt) / (len(docs)- long)}')
print(len(docs)-long)

new_data.append({"모델이 수정할 확률" : cnt / (len(docs)- long), "잘못 생성된 요약문을 모델이 잘 수정함" : good_cnt / (len(docs)- long),
                 "잘못 생성된 요약문을 모델이 잘 못 수정함": bad_cnt / (len(docs)- long),
                 "잘못 생성된 요약문을 모델이 수정 안함" : no_cnt / (len(docs)- long),
                 "잘 생성된 요약문을 모델이 잘못 수정함" : wrong_cnt / (len(docs)- long),
                 "잘 생성된 요약문을 모델이 수정안함" : v_cnt / (len(docs)- long),
                 "정확도" : (good_cnt + v_cnt) / (len(docs)- long),
                "총 기사 수" : len(docs)-long
                 })
save_data(new_data)
