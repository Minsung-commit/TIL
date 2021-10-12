# 필수 모듈 호출
import doc_classification as dc
import insta_prepro as ipp
import make_all as ma
import modeling as md
import naver_prepro as npp
import preprocessing_2 as pp
import pandas as pd
import numpy as np
import re
import time
from kiwipiepy import Kiwi
import tomotopy as tp


# 문헌별 토픽 분류

def doc_classification(insta_data, token_data, model):
  insta_data = pd.read_csv('tot_dataset.csv') #인스타 크롤링 데이터 원본 호출
  model = tp.LDAModel.load('TF-IDF_model.bin') #모델 호출
  all_items = pd.read_csv('all_items.csv') #전처리된 통합 데이터 호출 

  topic_table = pd.DataFrame() #테이블 생성
  doc_index = []
  content = []
  topics = []
  item_idx = []

  for i in range(len(model.docs)): # 인덱스별 값을 개별 변수로 테이블에 배정
    doc_index = int(i)
    content = all_items.content[i]
    item_idx = all_items.item_idx[i]
    topics = int(model.docs[i].get_topics(top_n=1)[0][0])
    topic_table = topic_table.append(pd.Series([doc_index,item_idx, content, topics]), ignore_index=True)

  topic_table.columns = ['doc_index','item_index', 'content', 'topic'] # 컬럼명 재정의

  topic_table = pd.DataFrame(topic_table) 

  insta_data.columns = ['item_index', 'content', 'date', 'like', 'tags', 'name', 'overlap', 'place'] # 컬럼명 재정의

  merged_data = pd.merge(topic_table, insta_data, on='item_index') #itme

  merged_data.drop('content_y', axis=1, inplace=True)

  return merged_data
