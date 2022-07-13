#import os
from shutil import ExecError
from bson import _encode_maxkey
from gensim.models import word2vec, Word2Vec
from numpy import size
import plotly.graph_objects as go
import plotly.io as po
from sklearn.decomposition import PCA
import matplotlib
import collections
import matplotlib.pyplot as plt
from konlpy.tag import Okt
import pandas as pd
import seaborn as sns


def Myword2vec(result):
    counts = collections.Counter(result)
    # print(counts)
    # print(counts.most_common(1)[0][0])

    keyword = counts.most_common(1)[0][0]

    fileName = '{}.txt'.format(keyword)
    with open(fileName, mode='w', encoding='utf-8') as fw:
        fw.write('/n'.join(result))

    genObj = []

    with open(fileName, mode='r', encoding='utf-8') as fr:
        line = fr.readline()
        okt = Okt()
        line = okt.nouns(line)
        print('line : ', line)
        genObj.append(line)
        print('genObj : ', genObj)

    # genObj = word2vec.LineSentence(fileName)
    print('genObj :', genObj)
    # word Embedding(단어를 수치화)의 일종으로 word2vec
    model = word2vec.Word2Vec(sentences=genObj, size=100,
                              window=10, min_count=20, sg=1)
    # ** word2vec 옵션 정리 **
    # vector_size : 벡터의 크기를 설정
    # window : 고려할 앞뒤 폭 (앞뒤의 단어를 설정)
    # min_count : 사용할 단어의 최소빈도 (설정된 빈도 이하 단어는 무시)
    # workers : 동시에 처리할 작업의 수
    # sg : 0 (CBOW), 1 (Skip-gram)
    # CBOW는 주변 단어를 통해 주어진 단어를 예측하는 방법
    # skip-gram은 하나의 단어에서 여러 단어를 예측하는 방법
    # 보편적으로 skip-gram이 많이 쓰인다.
    model.init_sims(replace=True)  # 필요없는 메모리 해제

    # 모델 저장
    model_path = 'static/model/' + keyword + '.model'
    try:
        model.save(model_path)
    except Exception as e:
        print('err : ', e)

    # 모델 불러오기
    model = word2vec.Word2Vec.load(model_path)

    # 사진 속의 단어와 가장 유사한 단어 추출 및 DataFrame으로 저장
    result = pd.DataFrame(model.wv.most_similar(
        keyword, topn=10), columns=['단어', '유사도'])

    # 시각화1
    sns.set(style='whitegrid', font='Malgun Gothic', font_scale=1)
    graph = sns.PairGrid(result.sort_values(
        '유사도', ascending=False), y_vars=['단어'])
    graph.fig.set_size_inches(5, 10)

    graph.map(sns.stripplot, size=10, orient='h',
              palette='ch:s=1, r=-1, h=1_r', linewidth=1, edgecolor='w')
    graph.set(xlim=(0.0, 1), xlabel='유사도', ylabel='')
    titles = keyword

    for ax, title in zip(graph.axes.flat, titles):
        ax.set(title=titles)
        ax.xaxis.grid(False)
        ax.yaxis.grid(True)    # 수직격자 False, 수평격자 True

        sns.despine(left=True, bottom=True)
    plt.savefig('static/word2vecimg/{}1.png'.format(keyword),
                bbox_inches='tight')  # 경로설정
    # plt.show()
    plt.close()

    # 시각화2
    word_vectors = model.wv
    vocabs = word_vectors.wv.vocab
    vocabs = list(vocabs.keys())
    word_vectors_list = [word_vectors[v] for v in vocabs]
    # print(word_vectors_list)

    pca = PCA(n_components=2)
    xys = pca.fit_transform(word_vectors_list)
    xs = xys[:, 0]
    ys = xys[:, 1]

    # plt.style.use(['dark_background'])
    s = 80
    plt.scatter(xs, ys, marker='o', alpha=0.7,
                edgecolors='w', cmap='Green', s=s)
    for i, v in enumerate(vocabs):
        plt.annotate(v, xy=(xs[i], ys[i]))
    plt.grid(False)

    plt.savefig('static/word2vecimg/{}2.png'.format(keyword),
                bbox_inches='tight')  # 경로설정
    # plt.show()
    plt.close()

    # 시각화3
    # pip install plotly
    fig = go.Figure(data=go.Scatter(
        x=xs, y=ys, mode='markers+text', text=vocabs, marker=dict(size=80)))
    fig.update_layout(template='plotly_dark')
    fig.update_layout(title=keyword)
    po.write_html(
        fig, file='templates/chart.html')  # 경로설정
    # fig.show()

    return keyword
