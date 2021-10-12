# Infer 새로운 문서 추론


## 필수 모듈 호출
import pandas as pd
import numpy as np
import re
import tomotopy as tp


def inference():
    ## 모델 호출
    best_model = tp.LDAModel.load('TF-IDF_model.bin')
    
    ## input 생성
    unseen_text = input('문서를 입력하세요')
    
    unseen_text = unseen_text.strip().split() #tokenizing
    
    doc_inst = best_model.make_doc(unseen_text)
    
    topic_dist, ll = best_model.infer(doc_inst, iter=200)

    return print("이 문서의 카테고리는 {}번입니다.".format(doc_inst.get_topics()[0][0]))

