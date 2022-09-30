import os
import torch
from PIL import Image
import numpy as np
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt


def yoloModel(image):
    best_data = 'static/model/best.pt'
    model = torch.hub.load('ultralytics/yolov5', 'custom',
                           path=best_data)
    ckpt = torch.load(best_data)['model']  # load checkpoint
    model.names = ckpt.names
    print('model.names : ', model.names)
    print('image : ', image)
    img1 = Image.open('static/uploads/'+str(image))

    # Inference
    results = model(img1, size=640)  # includes NMS
    results.save(str(image))

    files_Path = "runs/detect/"  # 파일들이 들어있는 폴더
    file_name_and_time_lst = []
    # 해당 경로에 있는 파일들의 생성시간을 함께 리스트로 넣어줌.
    for f_name in os.listdir(f"{files_Path}"):
        written_time = os.path.getctime(f"{files_Path}{f_name}")
        file_name_and_time_lst.append((f_name, written_time))
    # 생성시간 역순으로 정렬하고,
    sorted_file_lst = sorted(file_name_and_time_lst,
                             key=lambda x: x[1], reverse=True)
    # 가장 앞에 이는 놈을 넣어준다.
    recent_file = sorted_file_lst[0]
    recent_file_name = recent_file[0]
    #print('recent_file : ', recent_file)
    # print('recent_file_name : ', recent_file_name)  # exp 가장 최근 파일
    img2 = Image.open('runs/detect/'
                      + str(recent_file_name) + '/' + str(image))
    img2.save('static/yolo_img/'+str(image))

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
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    # ['병', '와인잔', '풍선', '캔', '빵', '케잌']
    print('0.7이상 weight 단어 한글화:', unique_list_kr)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    # print(word_list)
    # print(len(word_list))
    # results.show()  # or .show() .save()

    #import shutil
    # shutil.rmtree('D:/kim/acorn/work2/psou/final_project/myapp/static/crawling')

    # tokenized_doc = []
    # for i in range(2):
    #     tok_list = pd.read_csv('static/crawling/'+str(
    #         i)+'_results.csv')['0'].str.extract(r'([ㄱ-ㅎ|ㅏ-ㅣ|가-힣]+)', expand=False).dropna().tolist()
    #     tokenized_doc.append(tok_list)
    # return unique_list_kr, tokenized_doc, tok_list

    # 0.7이상 weight 단어 한글화에 대한 csv 파일만 가져와 워드투벡터화 함
    tokenized_doc = []
    unique_list_kr_test = ['케잌']
    for key in unique_list_kr:
        print("++++++++++++++++++++++++++++++++++++")
        print('key : ', key)
        print("++++++++++++++++++++++++++++++++++++")
        tok_list = pd.read_csv('static/crawling/'+str(
            key)+'_results.csv')['0'].str.extract(r'([ㄱ-ㅎ|ㅏ-ㅣ|가-힣]+)', expand=False).dropna().tolist()
        tokenized_doc.append(tok_list)
        #print("=====================================")
        #print('tokenized_doc : ', tokenized_doc)
        #print("=====================================")
    return unique_list_kr, tokenized_doc, tok_list
