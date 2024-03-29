#-*- coding:utf-8 -*-
# @Time : 2019/8/7
# @Author : Botao Fan
import pandas as pd
import numpy as np
import sys
import time
import multiprocessing


def cal_one(arr):
    if np.sum(arr) == 0:
        raise ValueError('Data do not contain positive point!')
    arr = np.array(arr)
    arr_cumsum = np.cumsum(arr)
    arr_pos = np.arange(len(arr)) + 1
    arr_result = (arr + 0.0) * arr_cumsum / arr_pos
    return (np.sum(arr_result)+0.0) / np.sum(arr)


def my_score(test_true, test_pred):
    '''
    :param result_true: DataFrame['user_id', 'job_id', 'delivered', 'satisfied']
    :param result_pred: DataFrame ['user_id', 'job_id', 'score']
    :return:
    '''
    result = pd.merge(left=test_pred, right=test_true, on=['user_id', 'job_id'], how='left')
    result.sort_values(['user_id', 'score'], ascending=False, inplace=True)
    score_delivered = result.groupby('user_id')['delivered'].apply(lambda x: cal_one(x))
    score_satisfied = result.groupby('user_id')['satisfied'].apply(lambda x: cal_one(x))
    score = result.groupby('user_id')[['user_id']].count()
    score['score_delivered'] = score_delivered
    score['score_satisfied'] = score_satisfied
    score['score'] = score['score_delivered'] * 0.3 + score['score_delivered'] * 0.7
    return score['score'].mean()


def pr(t_index):
    for i in range(40):
        print str(t_index) + '\t' + str(i)
        a = np.random.rand(200, 200)
        for i in range(4000):
            t = np.random.rand(200, 200)
            a *= t
        time.sleep(3)


def execute():
    process = []
    for i in range(4):
        t = multiprocessing.Process(target=pr, args=(i, ))
        t.start()
        process.append(t)

    for p in process:
        p.join()

if __name__ == '__main__':
    pass