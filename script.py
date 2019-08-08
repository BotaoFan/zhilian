#-*- coding:utf-8 -*-
# @Time : 2019/8/5
# @Author : Botao Fan
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import re

def set_format():
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.width', 1000)

def show_data_info_in_table(data):
    num_rows, num_columns = data.shape
    print '<table>'
    for i in range(num_columns):
        tmp = data.iloc[:2, i]
        print '<tr><td>' + str(tmp.name) + '</td><td> </td><td>' + str(tmp.dtype) + '</td>'
    print '</table>'


def split_str(data, sep='[/(), |]'):
    return data.apply(lambda x: set(re.split(sep, x)))



def explory_cate_data(data, col_name, show_min_count=10):
    print 'Dtype of %s is %s and number of categoris is %d' % \
          (col_name, data[col_name].dtype, len(data[col_name].unique()))
    print 'Category of %s are:' % col_name
    cate = data[col_name].unique()
    cate.sort()
    print cate
    print 'Count of every category'
    cate_count = data.groupby(col_name)[[col_name]].count()
    cate_count.columns = ['count']
    cate_count['perc'] = cate_count['count']/(cate_count['count'].sum()+0.0)
    cate_count.sort_values('count', ascending=False, inplace=True)
    print cate_count[cate_count['count'] >= show_min_count]
    return cate_count


if __name__ == "__main__":
    set_format()
    data_path = '/Users/botaofan/PycharmProjects/tianchi/zhilian/data/'
    #======Load Data======
    #load train_user
    raw_user_dtype = {'live_city_id': np.int16, 'desire_jd_salary_id': object, 'cur_salary_id': object, 'birthday': np.int16,
                      'start_work_date': object, }

    raw_user = pd.read_csv(data_path + 'table1_user', delimiter='\t', error_bad_lines=False, dtype=raw_user_dtype)
    #load train_job
    raw_job_dtype = {'city': np.int16, 'require_nums': np.int16, 'max_salary': object, 'min_salary': object,
                     'start_date': object, 'end_date': object, 'raw_job': np.int16, 'is_travel': np.int16,
                     'min_years': np.int16, }
    raw_job = pd.read_csv(data_path + 'table2_jd', delimiter='\t', error_bad_lines=False, dtype=raw_job_dtype)
    #load train_action
    raw_action_dtype = {'browsed': np.int16, 'delivered': np.int16, 'satisfied': np.int16}
    raw_action = pd.read_csv(data_path + 'table3_action', delimiter='\t', error_bad_lines=False, dtype=raw_action_dtype)
    #load test_user
    test_user_dtype = {'live_city_id': np.int16, 'desire_jd_salary_id': object, 'cur_salary_id': object, 'birthday': np.int16,
                       'start_work_date': object}
    test_user = pd.read_csv(data_path + "user_ToBePredicted", delimiter="\t", error_bad_lines=False, dtype=test_user_dtype)
    #load test_action
    test_action = pd.read_csv(data_path + "zhaopin_round1_user_exposure_A_20190723", delim_whitespace=True)

    user_action = raw_action.groupby(['user_id', 'jd_no'])['delivered', 'satisfied'].max().reset_index()
    #======Data Exploration and Preprocess======
    tmp = raw_user['desire_jd_city_id'].str.split(',')
    raw_user['desire_jd_city_id_0'] = tmp.apply(lambda x: x[0])
    raw_user['desire_jd_city_id_1'] = tmp.apply(lambda x: x[1])
    raw_user['desire_jd_city_id_2'] = tmp.apply(lambda x: x[2])

    raw_user['desire_jd_industry_set'] = raw_user['desire_jd_industry_id'].apply(lambda x: set(x.split('/')))


