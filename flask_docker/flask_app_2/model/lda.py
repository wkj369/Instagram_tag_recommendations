#import os
import json
from xgboost.training import cv
from gensim.test.utils import datapath
from gensim import corpora
import gensim
import pandas as pd


def lda_model(keyword, tokenized_doc):
    # 2. gensim을 통해 Corpus(말뭉치) Dictionary(사전) 언어모델 형성
    dictionary = corpora.Dictionary(tokenized_doc)
    corpus = [dictionary.doc2bow(text) for text in tokenized_doc]

    # 3. 적절한 토픽수를 결정하기 위해  Perplexity 및 Coherence 계산
    # 3-1. Perplexity 및 Coherence 계산 함수 정의
    def compute_metrics_values(dictionary, corpus, limit, start=2, step=3):
        coherence_values = []
        perplexity_values = []
        model_list = []
        for num_topics in range(start, limit, step):
            model = gensim.models.ldamodel.LdaModel(
                corpus, num_topics=num_topics, id2word=dictionary, passes=15)
            model_list.append(model)
            coherence_model_lda = gensim.models.CoherenceModel(model=model,
                                                               corpus=corpus, dictionary=dictionary, coherence='u_mass')
            coherence_values.append(coherence_model_lda.get_coherence())
            perplexity_values.append(model.log_perplexity(corpus))
        return model_list, coherence_values, perplexity_values

    # 3-2. compute_metrics_values함수 호출
    model_list, coherence_values, perplexity_values = \
        compute_metrics_values(dictionary=dictionary,
                               corpus=corpus, start=2, limit=8, step=1)

    # 3-3. Perplexity 및 Coherence 계산 결과 출력
    limit = 8
    start = 2
    step = 1
    x = range(start, limit, step)

    dic = {}
    for m, cv in zip(x, coherence_values):
        print("Num Topics =", m, " has Coherence Value of", round(cv, 4))

    for m, pv in zip(x, perplexity_values):
        print("Num Topics =", m, " has perplexity values of", round(pv, 4))

        dic[abs(round(pv, 4))] = m

    # 4. Perplexity가 가장 낮은 토픽수 지정
    a = min(dic.keys())

    # 5. LDA 모델
    # 5-1. LDA모델 생성
    file = 'static/model/lda.model'
    NUM_TOPICS = dic[a]  # dic[a]개의 토픽
    lda = gensim.models.ldamodel.LdaModel(
        corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=15, random_state=121)

    # 5-2. LDA모델 저장
    lda_model_path = 'static/model/lda.model'
    lda.save(lda_model_path)

    # 5-3. LDA모델 불러오기
    # fname = datapath(
    #     "static/model/lda.model")
    lda = gensim.models.LdaModel.load(lda_model_path, mmap='r')

    # 6. 학습시킨 모델의 각 토픽에 할당된 단어 리스트
    word = gensim.models.coherencemodel.CoherenceModel.top_topics_as_word_lists(
        lda, dictionary, topn=30)

    # 7. 토픽 인덱스와 각 토픽에 할당된 단어들의 가중치
    topics = lda.print_topics(num_words=30)
    for topic in topics:
        print(topic)

    # 8. 학습한 모델에 대한 평가 지표 Perplexity 및 Coherence
    # 8-1. Perplexity Score: a measure of how good the model is. lower the better.
    print('\nPerplexity: ', lda.log_perplexity(corpus))

    # 8-2. Coherence Score
    coherence_model_lda = gensim.models.CoherenceModel(
        model=lda, corpus=corpus, dictionary=dictionary, coherence='u_mass')
    coherence_lda = coherence_model_lda.get_coherence()
    print('Coherence Score: ', coherence_lda)

    # 9. 말뭉치의 단어들 출현 빈도수 계산
    dict = {}
    for cp in corpus:
        for id, freq in cp:
            dict[dictionary[id]] = freq
    #print('dict:', dict)

    # 10. 학습 후 나온단어들의 출현 빈도수 계산
    qq = list(dict.keys())
    ddd = []
    for q in qq:
        for i in range(len(word)):
            for w in word[i]:
                if q == w:

                    ddd.append([i, (dict[q], q)])
                    #print([i, (q, dict[q])])
    # print(ddd)

    # 11. pandas의 DataFrame 형식으로 빈도수가 높은 단어 순으로 주제별로 저장
    df = pd.DataFrame(columns=list(range(len(word))))

    for i in df.columns:
        for j in range(len(ddd)):
            if i == ddd[j][0]:
                df.loc[j, i] = ddd[j][1]

    l = []
    for i in df.columns:
        for j in range(30):
            l.append(j)

    df['인덱스'] = l
    df = df.set_index('인덱스')

    se_li = []
    for i in df.columns:
        se_li.append(df.loc[:, i].dropna().sort_values(
            ascending=False).reset_index().drop(['인덱스'], axis=1))

    re_l = []
    for j in range(30):
        re_l.append(j)

    df = pd.concat(se_li, axis=1)
    print(df)

    # 12. 웹 화면 상에 관련 단어 시각화를 위해 json 형식으로 변환
    jli = []
    jlis = []

    json_dic = {'name': keyword,
                "children": jli}

    for i in range(len(df.columns)):
        jli2 = []
        for j in range(5):
            jli2.append({'name': df.iloc[j+1, i][1],
                        'value': df.iloc[j+1, i][0]})
        jlis.append(jli2)
        jli.append({'name': df.iloc[0, i][1], 'children': jlis[i]})

    lda_data = json.dumps(json_dic, ensure_ascii=False)

    return lda_data
