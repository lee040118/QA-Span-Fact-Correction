# QA-Span-Fact-Correction

#### Paper: [Multi-Fact Correction in Abstractive Text Summarization](https://arxiv.org/abs/2010.02443)

## Overview
- 문서 요약
  - Source text x가 주어졌을 때 target text y를 작성하는 것
  - text y는 짧아야 하며, source text x의 중요한 정보를 포함 해야함

- Abstractive Summarization
  - 원 문서에 존재하거나 또는 모델이 직접 단어를 생성해 요약문 작성
    - 장점) 원문에 없는 단어라도 상황에 맞게 적절히 바꾸어 유연한 문장을 생성
    - 단점) 부정확성 -> 사실 불일치 문제 (추상 요약 모델이 생성하는 요약문이 본문 내용과 일치 하지 않는 문제)
    - 요약 모델이 정보 제공 뿐만 아니라, 정보의 정확성에 대해서 최적화 필요

- Inspired by QA?
  - QA? Question에 정답이 되는 Paragraph의 substring을 뽑아내는 것(start/end Span을 찾아내는 task)
  - 요약문에서 잘못 생성된 개체명에 대해 QA모델을 통해 정답 개체명을 뽑아내 교체하는 방식
  - 한번에 하나의 개체명을 masked하고 교체
  
## Requirements
```
pytorch==1.7.1
transformers==4.3.3
```
## Data
- [Dacon 한국어 문서 생성요약 AI 경진대회](https://dacon.io/competitions/official/235673/overview/) 의 학습 데이터 + 국립 국어원 Data set
- 기사 원문 context, 개체명이 마스킹된 요약문이 question으로 주어짐
- 정답 개체명이 여러 개인 경우 위치 선정 기준?
  - 개체명이 포함된 본문의 문장과 요약 문장간의 유사도를 통해 정답 선정(자카드 유사도)


- Data 구조
  - paragraphs
    - qas 
      - answer : [text, answer_start]
      - id : (기사 ID) - (질문 번호)
      - question : 정답 개체명이 Masking된 요약문
    - context : 기사 본문
  - title : (기사 ID)


```
sh Create_data.sh
```   
 
 ## Model architecture
 <img src="model.PNG" width="700">
 
 ## How to Train
 - KoELECTRA summarization fine-tuning
 - Finetunig에는 [KoELECTRA](https://github.com/monologg/KoELECTRA)의 discriminator를 사용
```
python3 run_qa.py --task korquad --config_file koelectra-small-v3.json
```   

## Eval in pororo summary
![image](https://user-images.githubusercontent.com/69192178/125232854-79f8c980-e318-11eb-9c3d-2ce2dfb74fca.png)
![image](https://user-images.githubusercontent.com/69192178/125232919-94cb3e00-e318-11eb-879b-e3c6a3ae638f.png)
![image](https://user-images.githubusercontent.com/69192178/125232968-a7de0e00-e318-11eb-8ac4-b137858aeed6.png)

### Example
![image](https://user-images.githubusercontent.com/69192178/125233022-bf1cfb80-e318-11eb-96dd-56d64615635b.png)



