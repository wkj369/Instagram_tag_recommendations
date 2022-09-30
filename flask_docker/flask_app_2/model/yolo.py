#import os
import torch
from PIL import Image
import numpy as np
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt


def yoloModel(image):
    best_data = 'static/model/best.pt'
    #model = torch.load(best_data)
    model = torch.hub.load('ultralytics/yolov5', 'custom',
                           path=best_data)
    ckpt = torch.load(best_data)['model']  # load checkpoint
    model.names = ckpt.names
    print('model.names : ', model.names)
    print('image : ', image)
    img1 = Image.open('static/uploads/'+str(image))

    # Inference
    results = model(img1, size=640)  # includes NMS
    results.save('static/yolo_img/')

    # Results
    results.print()
    print('results.names : ', results.names)

    word_list = []

    # print('pred')
    # print(results.pred)
    # print('xyxy[0]')
    # print(results.xyxy[0])

    # 과학적 표기법 대신 소수점 6자리까지 나타낸다.
    np.set_printoptions(precision=6, suppress=True)
    weight_list = []
    for i, det in enumerate(results.pred):
        # print(i)
        # print(det)
        # Print results
        for c in det[:, -1].unique():
            # print(int(c))
            word_list.append((results.names[int(c)]))

        for *xyxy, conf, cls in reversed(det):
            label = '%s %.2f' % (results.names[int(cls)], conf)
            if round(float(conf), 2) >= 0.7:
                weight_list.append((results.names[int(cls)]))

    # print(weight_list)
    unique_set = set(weight_list)
    unique_list = list(unique_set)
    print('0.7이상 weight 단어:', unique_list)

    unique_list_kr = []
    for word in unique_list:
        if word == 'balloons':
            change_kr = word.replace("balloons", "풍선")

        if word == 'bottle':
            change_kr = word.replace("bottle", "병")

        if word == 'bread':
            change_kr = word.replace("bread", "빵")

        if word == 'bus':
            change_kr = word.replace("bus", "버스")

        if word == 'cake':
            change_kr = word.replace("cake", "케잌")

        if word == 'can':
            change_kr = word.replace("can", "캔")

        if word == 'cookie':
            change_kr = word.replace("cookie", "쿠키")

        if word == 'melon':
            change_kr = word.replace("melon", "멜론")

        if word == 'palm tree':
            change_kr = word.replace("palm tree", "야자수")

        if word == 'person':
            change_kr = word.replace("person", "사람")

        if word == 'pineapple':
            change_kr = word.replace("pineapple", "파인애플")

        if word == 'wine glass':
            change_kr = word.replace("wine glass", "와인잔")

        if word == 'car':
            change_kr = word.replace("car", "자동차")

        unique_list_kr.append(change_kr)

    print('0.7이상 weight 단어 한글화:', unique_list_kr)

    # print(word_list)
    # print(len(word_list))
    # results.show()  # or .show() .save()

    #import shutil
    # shutil.rmtree('D:/kim/acorn/work2/psou/final_project/myapp/static/crawling')

    tokenized_doc = []
    for i in range(4):
        tok_list = pd.read_csv('static/crawling/'+str(
            i)+'_results.csv')['0'].str.extract(r'([ㄱ-ㅎ|ㅏ-ㅣ|가-힣]+)', expand=False).dropna().tolist()
        tokenized_doc.append(tok_list)
    return unique_list_kr, tokenized_doc, tok_list
