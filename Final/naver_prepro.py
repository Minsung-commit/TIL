## naver 블로그 데이터 전처리
# 입력값 : 전처리 되기 전 네이버 크롤링데이터(csv파일 형식)
# 출력값 : 전처리 된 네이버 데이터

def naver_preprocessing(naver):

    #데이터 로드
    naver_data = pd.read_csv(naver)

    # 인덱스 제거
    naver_data.drop('Unnamed: 0', axis=1, inplace=True)

    # 토큰화
    for i in range(len(naver_data)):
        naver_data.content[i] = pp.sub_special_token(naver_data.content[i])

    #클린징
    pp.word_cleansing(naver_data)

    return naver_data