

# make_total_list
## 입력값 : 전처리된 네이버, 인스타 데이터
## 출력값 : 병합된 네이버,인스타 데이터

def make_total_list(naver_data, insta_data):
    n_data = pd.DataFrame(naver_data.copy()) # 네이버 블로그 데이터 로드(콘텐츠)
    i_data = pd.DataFrame(insta_data[['Unnamed: 0', 'content']].copy()) # 인스타 데이터 로드(인덱스, 콘텐츠)

    i_data.columns = ['item_idx', 'content'] # merge를 위한 식별자 컬럼('item_idx') 생성

    all_items = pd.concat([i_data, n_content]) # 두 데이터 행 병합

    all_items.reset_index(drop=True, inplace=True) #인덱스 리셋

    #빈 샘플 확인 및 제거

    drop = [index for index in range(len(all_items)) if len(all_items.content[index]) < 1] # drop에 empty 샘플의 인덱스를 저장


    while len(drop) > 0: 

        # 빈 샘플 제거
        for i in drop:
            all_items.drop(all_items.index[i], inplace=True)
        # 인덱스 리셋
        all_items.reset_index(drop=True, inplace=True)

        drop = [index for index in range(len(all_items)) if len(all_items.content[index]) < 1] # 재확인 

     # 인덱스 리셋
    all_items.reset_index(drop=True, inplace=True) #최종 리셋

    return all_items