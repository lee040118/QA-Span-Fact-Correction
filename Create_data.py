import argparse
import json
import os
import copy
from tqdm import tqdm
from pororo import Pororo
from collections import defaultdict
from transformers import ElectraTokenizer
tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

def Jaccard(doc1, doc2):
    a = set(tokenizer.tokenize(doc1))
    b = set(tokenizer.tokenize(doc2))
    c= a.intersection(b)
    return float(len(c)) / (len(a)+len(b)-len(c))

def load_jsonl(file_path):
    with open(file_path, encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data

 def load_json(file_path):
    return json.load(open(file_path, 'r'))
  
def save_data_jsonl(args, data):
    output_file = os.path.splitext(args.source_file)[0] + "_qa.jsonl"
    with open(output_file, "w", encoding="utf-8") as fd:
        for example in tqdm(data):
            example = dict(example)
            if 'label' not in example:
                example['label'] = 'CORRECT'
            fd.write(json.dumps(example, ensure_ascii=False) + "\n")
            
def save_data_json(args, data):
    output_file = os.path.splitext(args.source_file)[0] + "_cate.json"
    with open(output_file, "w", encoding="utf-8") as fd:
        fd.write(json.dumps(data, ensure_ascii=False))
        
def ko_ner(args, data):
    ner = Pororo(task="ner", lang="ko")
    for example in tqdm(data):
        ner_text = defaultdict(list)
        ner_summary = defaultdict(list)
        tmp = []
        lang_txt = ""
        stack = []
        for idx, txt in enumerate(example['text']):
            if (txt == "「" or txt == '《' or txt == '[' or txt == '(' or txt == "\'" or txt == '‘' or txt == '“') and not stack:
                stack.append(txt)
                if idx != 0 and example['text'][idx - 1] == ' ':
                    tmp += [(' ', 'O')]
                if not lang_txt:
                    lang_txt = txt
                    continue
                else:
                    tmp += ner(lang_txt)
                lang_txt = txt
            elif (txt == "「" or txt == '《' or txt == '[' or txt == '(' or txt == '‘' or txt == '“') and stack:
                stack.append(txt)
                lang_txt += txt

            elif (txt == '》' or txt == ']' or txt == "\'" or txt == ')' or txt == '’' or txt == '”') and stack:
                if txt == "\'" and "\'" in stack:
                    stack.remove(txt)
                elif txt == '」' and '「' in stack:
                    stack.remove('「')
                elif txt == '」' and '《' in stack:
                    stack.remove('《')
                elif txt == ')' and '(' in stack:
                    stack.remove('(')
                elif txt == ']' and '[' in stack:
                    stack.remove('[')
                elif txt == '’' and '‘' in stack:
                    stack.remove('‘')
                elif txt == '”' and '“' in stack:
                    stack.remove('“')
                lang_txt += txt
                if not stack:
                    tmp += ner(lang_txt)
                    if idx != len(example['text']) - 1 and example['text'][idx + 1] == ' ':
                        tmp += [(' ', 'O')]
                    lang_txt = ""
                else:
                    stack.append(txt)

            elif len(lang_txt) > 500 or idx == len(example['text']) - 1:
                lang_txt += txt
                tmp += ner(lang_txt)
                if txt == ' ':
                    tmp += [(' ', 'O')]
                if idx != len(example['text']) - 1 and example['text'][idx + 1] == ' ':
                    tmp += [(' ', 'O')]

                lang_txt = ""
            else:
                lang_txt += txt

        text_st = 0
        for tar, ent in tmp:
            if ent == 'O':
                text_st += len(tar)
                continue
            ner_text[ent].append((text_st, tar))
            text_st += len(tar)
        example["text_ner"] = ner_text

        summary_st = 0
        tmp = ner(example["summary"])
        for tar, ent in tmp:
            if ent == 'O':
                summary_st += len(tar)
                continue
            ner_summary[ent].append((summary_st, tar))
            summary_st += len(tar)
        example["summary_ner"] = ner_summary

    return data
  
def create_span(data):
    new_example = {"data" : []}
    d_idx = -1
    for example in tqdm(data):
        categories = ("PERSON","LOCATION","ORGANIZATION","CIVILIZATION" ,"DATE", "TIME","QUANTITY")
        id, text, summary, text_ner, summary_ner, label = example["id"], example["text"], example["summary"], example["text_ner"], example["summary_ner"], example["label"]
        tok_len = len(tokenizer.tokenize(summary))
        if tok_len > 64:
            continue
        summary_ents = [ent for ent in summary_ner.keys() if ent in categories]
        text_ents = [ent for ent in text_ner.keys() if ent in categories] # text에 있는 개체명 종류들
        if label == "INCORRECT" or len(summary_ents) == 0 :
            continue
        d_idx += 1
        cnt = -1
        tmp = {"paragraphs": [{"qas": [{"answers": [], }]}]}
        new_example["data"].append(copy.deepcopy(tmp))
        new_example["data"][d_idx]["title"] = id
        new_example["data"][d_idx]["paragraphs"][0]["context"] = text
        for ents in summary_ents:
            for spoint, ent in summary_ner[ents]: # ents : 개체명 종류, ent : 실제 개체명 이름

                if not text_ner.get(ents):
                    continue
                answer_start = -1
                maxsimilarity = 0
                for pos, ans in text_ner[ents]:
                    if ans == ent:
                        doc = text[:pos].split('. ')[-1] + text[pos:].split('. ')[0]
                        similarity = Jaccard(doc, summary)
                        if similarity > maxsimilarity:
                            maxsimilarity = similarity
                            answer_start, answer_end = pos, pos+len(ent) -1


                sp, ep = spoint, spoint + len(ent)
                mask_sum = summary[:sp] + "[MASK]" + summary[ep:]

                if answer_start != -1 and sp != -1:
                    cnt+=1
                    if cnt == 0:
                        new_example["data"][d_idx]["paragraphs"][0]["qas"][0]["answers"].append(
                            {"text": ent, "answer_start": answer_start})
                        new_example["data"][d_idx]["paragraphs"][0]["qas"][0]["id"] = id + "-" + str(cnt)
                        new_example["data"][d_idx]["paragraphs"][0]["qas"][0]["question"] = mask_sum
                    else :
                        new_example["data"][d_idx]["paragraphs"][0]["qas"].append({"answers": [{"text": ent, "answer_start": answer_start}], "id" : id+"-" + str(cnt), "question" : mask_sum})

        if cnt == -1:
            new_example["data"].pop()
            d_idx -=1

    return new_example


def main(args):
    data = load_json(args.source_file)
    print("Ko_Named Entity Recognition")
    data = ko_ner(args, data)
    save_data_jsonl(args, data)
    
    data = load_jsonl(args.source_file2)
    span_data = create_span(data)
    save_data(args,span_data)

if __name__ == "__main__":
        PARSER = argparse.ArgumentParser()
        PARSER.add_argument("--source_file", type=str, help="Path to file contains source documents.")
        PARSER.add_argument("--source_file2", type=str, help="Path to file contains source2 documents.")
        ARGS = PARSER.parse_args()

        main(ARGS)
