#-*- coding:utf-8 -*-
# @Time : 2019/8/5
# @Author : Botao Fan
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import xgboost as xgb
from datetime import datetime
import xgboost as xgb
import lightgbm as lgb
import warnings
import re
warnings.filterwarnings('ignore')

salary_dict = {'0000000000': np.nan, '0000001000': 1, '0100002000': 2, '0200104000': 3, '0400106000': 4,
               '0600108000': 5, '0800110000': 6, '1000115000': 7, '1500120000': 8, '1500125000': 9, '2000130000': 10,
               '2500199999': 11, '2500135000': 11, '3000150000': 12, '3500150000': 12, '5000170000': 13,
               '70001100000': 14, '100001150000': 15, '-': np.nan}
salary_max_dict = {'0000000000': np.nan, '0000001000': 1000, '0100002000': 2000, '0200104000': 4000, '0400106000': 6000,
                   '0600108000': 8000, '0800110000': 10000, '1000115000': 15000, '1500120000': 20000,
                   '1500125000': 25000, '2000130000': 30000,
                   '2500199999': 35000, '2500135000': 35000, '3000150000': 50000, '3500150000': 50000,
                   '5000170000': 70000,
                   '70001100000': 100000, '100001150000': 150000, '-': np.nan}
salary_min_dict = {'0000000000': np.nan, '0000001000': 0, '0100002000': 1000, '0200104000': 2000, '0400106000': 4000,
                   '0600108000': 6000, '0800110000': 8000, '1000115000': 10000, '1500120000': 15000,
                   '1500125000': 15000, '2000130000': 20000,
                   '2500199999': 25000, '2500135000': 25000, '3000150000': 30000, '3500150000': 35000,
                   '5000170000': 50000,
                   '70001100000': 70000, '100001150000': 100000, '-': np.nan}
degress_dict = {'初中': 1, '中专': 2, '中技': 2, '高中': 2, '大专': 3, '\N': 0, '请选择': 0,
                '本科': 4, '硕士': 5, 'MBA': 5, 'EMBA': 5, '博士': 6, '其他': np.nan, }
job_min_year_dict = {-1: 0, 103: 1, 305: 3, 510: 5, 1099: 10, 1: 1, 0: 0, 399: 3, 599: 5, 299: 2, 199: 1, 110: 1}


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


def split_str(x, sep='/'):
    return x if pd.isna(x) else set(x.split(sep))



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

#======Data clean========
def clean_user(raw_user):
    user = raw_user.copy()
    user.set_index('user_id', inplace=True)
    tmp = user['desire_jd_city_id'].str.split(',')
    user['desire_jd_city_id_0'] = tmp.apply(lambda x: x[0])
    user['desire_jd_city_id_1'] = tmp.apply(lambda x: x[1])
    user['desire_jd_city_id_2'] = tmp.apply(lambda x: x[2])
    user['desire_jd_industry_set'] = user['desire_jd_industry_id'].apply(split_str)
    user['desire_jd_type_set'] = user['desire_jd_type_id'].apply(split_str)
    user['desire_jd_max_salary'] = user['desire_jd_salary_id'].apply(lambda x: salary_max_dict[x])
    user['desire_jd_min_salary'] = user['desire_jd_salary_id'].apply(lambda x: salary_min_dict[x])
    user['desire_jd_salary_id'] = user['desire_jd_salary_id'].apply(lambda x: salary_dict[x])
    user['cur_industry_set'] = user['cur_industry_id'].apply(split_str)
    user['cur_jd_type_set'] = user['cur_jd_type'].apply(split_str)
    user['cur_salary_id'] = user['cur_salary_id'].apply(lambda x: salary_dict[x])
    user['cur_degree_id'] = user['cur_degree_id'].apply(lambda x: np.nan if pd.isna(x) else degress_dict[x.strip()])
    user['start_work_date'] = user['start_work_date'].apply(lambda x: np.nan if x == '-' else int(x))
    user['experience_set'] = user['experience'].apply(split_str, **{'sep': '|'})
    return user


def clean_job(raw_job):
    job = raw_job.copy()
    job['jd_title_set'] = job['jd_title'].apply(split_str)
    job.drop(columns=['company_name'], inplace=True)
    job['jd_sub_type_set'] = job['jd_sub_type'].apply(split_str)
    date_nan = '18000101'
    date_nan_datetime = pd.to_datetime(date_nan)
    job.loc[job['start_date'] == '\\N', 'start_date'] = date_nan
    job['start_date'] = pd.to_datetime(job['start_date'])
    job.loc[job['start_date'] == date_nan_datetime, 'start_date'] = np.nan
    job.loc[job['end_date'] == '\\N', 'end_date'] = date_nan
    job['end_date'] = pd.to_datetime(job['end_date'])
    job.loc[job['end_date'] == date_nan_datetime, 'end_date'] = np.nan
    job.loc[job['is_travel'] == 2, 'is_travel'] = np.nan
    job['min_years'] = job['min_years'].apply(lambda x: job_min_year_dict[x])
    job['key_set'] = job['key'].apply(split_str, **{'sep': '|'})
    job['min_edu_level'] = job['min_edu_level'].apply(lambda x: np.nan if pd.isna(x) else degress_dict[x.strip()])
    job.drop(columns=['max_edu_level', 'is_mangerial', 'resume_language_required'], inplace=True)
    job.set_index('jd_no', inplace=True)
    return job


def train_action_generate(raw_action):
    '''
    :param raw_action: DataFrame ['user_id', 'jd_no', 'delivered', 'satisfied']
    :return: DataFrame ['y', 'delivered', 'satisfied'], index = ['user_id', 'jd_no']
    '''
    action = raw_action.groupby(['user_id', 'jd_no'])[['delivered', 'satisfied']].max()
    action.reset_index(inplace=True)
    action['y'] = action['delivered'] * 0.3 + action['satisfied'] * 0.7
    action.set_index(['user_id', 'jd_no'], inplace=True)
    return action


def test_action_generate(raw_action):
    return raw_action.copy()


def find_len_set(x):
    try:
        l = len(x)
    except:
        l = 0
    return l


def feats_generate(action, user, job):
    #Get Training Data
    action = pd.merge(action, user, left_index=True, right_index=True, how='left')
    action = pd.merge(action, job, left_index=True, right_index=True, how='left')
    if 'y' in action.columns:
        action_feats = action[['y', 'delivered', 'satisfied']]
    else:
        action_feats = action[[]]
    #Features Generate
    big_city_dict = defaultdict(lambda: 0)
    big_city_dict['530'] = 1; big_city_dict['801'] = 1; big_city_dict['538'] = 1; big_city_dict['719'] = 1; big_city_dict['854'] = 1
    action_feats['feat1_live_city_is_big'] = action['live_city_id'].apply(lambda x: big_city_dict[x])
    action_feats['feat2_live_city_desire_0'] = (action['live_city_id'] == action['desire_jd_city_id_0'])*1
    action_feats['feat3_live_city_desire_1'] = (action['live_city_id'] == action['desire_jd_city_id_1'])*1
    action_feats['feat4_live_city_desire_2'] = (action['live_city_id'] == action['desire_jd_city_id_2'])*1
    action_feats['feat5_live_ctiy_desire'] = ((action_feats['feat2_live_city_desire_0'] + \
                                             action_feats['feat3_live_city_desire_1'] + \
                                             action_feats['feat4_live_city_desire_2']) >= 1)*1

    action_feats['feat6_city_live_job'] = (action['live_city_id'] == action['city'])*1
    action_feats['feat7_job_city_desire_0'] = (action['city'] == action['desire_jd_city_id_0'])*1
    action_feats['feat8_job_city_desire_1'] = (action['city'] == action['desire_jd_city_id_1'])*1
    action_feats['feat9_job_city_desire_2'] = (action['city'] == action['desire_jd_city_id_2'])*1
    action_feats['feat10_job_ctiy_desire'] = ((action_feats['feat7_job_city_desire_0'] + \
                                             action_feats['feat8_job_city_desire_1'] + \
                                             action_feats['feat9_job_city_desire_2']) >= 1)*1
    action_feats['feat11_desire_cur_indu_len'] = (action['desire_jd_industry_set']-(action['desire_jd_industry_set']-action['cur_industry_set'])).apply(find_len_set)
    action_feats['feat12_desire_cur_type_len'] = (action['desire_jd_type_set']-(action['desire_jd_type_set']-action['cur_jd_type_set'])).apply(find_len_set)
    action_feats['feat13_desire_job_type_len'] = (action['desire_jd_type_set']-(action['desire_jd_type_set']-action['jd_sub_type_set'])).apply(find_len_set)
    action_feats['feat14_desire_indu_job_key_len'] = (action['desire_jd_industry_set']-(action['desire_jd_industry_set']-action['key_set'])).apply(find_len_set)
    action_feats['feat15_desire_type_job_key_len'] = (action['desire_jd_type_set']-(action['desire_jd_type_set']-action['key_set'])).apply(find_len_set)

    action_feats['feat16_desire_cur_salary'] = action['desire_jd_salary_id'] - action['cur_salary_id']
    action_feats['feat17_desire_job_max_salary'] = action['desire_jd_max_salary'] - action['max_salary']
    action_feats['feat18_desire_job_min_salary'] = action['desire_jd_min_salary'] - action['min_salary']
    action_feats['feat19_birthday'] = action['birthday']
    action_feats['feat20_work_year'] = 2019 - action['start_work_date']
    action_feats['feat21_start_date_duration'] = (datetime(2019, 9, 1) - action['start_date']).apply(lambda x: x.days)
    action_feats['feat22_end_date_duration'] = (datetime(2019, 9, 1) - action['end_date']).apply(lambda x: x.days)
    action_feats['feat23_end_start_date_duration'] = (action['end_date'] - action['start_date']).apply(lambda x: x.days)
    action_feats['feat24_degree_min_edu'] = action['cur_degree_id'] - action['min_edu_level']
    action_feats['feat25_degree'] = action['cur_degree_id']
    action_feats['feat26_desire_jd_city_count'] = (action['desire_jd_city_id_0'] != '-')*1 +  \
                                           (action['desire_jd_city_id_1'] != '-')*1 + \
                                           (action['desire_jd_city_id_2'] != '-')*1
    action_feats['feat27_desire_jd_industry_count'] = action['desire_jd_industry_set'].apply(find_len_set)
    action_feats['feat28_desire_jd_type_count'] = action['desire_jd_type_set'].apply(find_len_set)
    action_feats['feat29_cur_salary_id'] = action['cur_salary_id']
    action_feats['feat30_cur_industry_count'] = action['cur_industry_set'].apply(find_len_set)
    action_feats['feat31_cur_jd_type_count'] = action['cur_jd_type_set'].apply(find_len_set)
    action_feats['feat32_jd_sub_type_count'] = action['jd_sub_type'].apply(find_len_set)
    action_feats['feat33_require_nums'] = action['require_nums']
    return action_feats




if __name__ == "__main__":
    set_format()
    data_path = '/Users/fan/PycharmProjects/tianchi/zhilian/data/'
    #======Load Data======
    #load train_user
    raw_user_dtype = {'live_city_id': object, 'desire_jd_salary_id': object, 'cur_salary_id': object, 'birthday': np.int16,
                      'start_work_date': object, }

    raw_user = pd.read_csv(data_path + 'table1_user', delimiter='\t', error_bad_lines=False, dtype=raw_user_dtype)
    #load train_job
    raw_job_dtype = {'city': object, 'require_nums': np.int16, 'max_salary': np.int32, 'min_salary': np.int32,
                     'start_date': object, 'end_date': object, 'raw_job': np.int16, 'is_travel': np.int16,
                     'min_years': np.int16, }
    raw_job = pd.read_csv(data_path + 'table2_jd', delimiter='\t', error_bad_lines=False, dtype=raw_job_dtype)
    #load train_action
    raw_action_dtype = {'browsed': np.int16, 'delivered': np.int16, 'satisfied': np.int16,}
    raw_action = pd.read_csv(data_path + 'table3_action', delimiter='\t', error_bad_lines=False, dtype=raw_action_dtype)
    #load test_user
    test_user_dtype = {'live_city_id': np.int16, 'desire_jd_salary_id': object, 'cur_salary_id': object, 'birthday': np.int16,
                       'start_work_date': object}
    test_user = pd.read_csv(data_path + "user_ToBePredicted", delimiter="\t", error_bad_lines=False, dtype=test_user_dtype)
    #load test_action
    test_action = pd.read_csv(data_path + "zhaopin_round1_user_exposure_A_20190723", delim_whitespace=True)
    user_action = raw_action.groupby(['user_id', 'jd_no'])['delivered', 'satisfied'].max().reset_index()
    #======Data Preprocess======
    user = clean_user(raw_user)
    job = clean_job(raw_job)
    action = a

    y = action_feats[['y']].values
    x = action_feats.drop(columns=['y', 'delivered', 'satisfied']).values

    model = xgb.XGBRegressor()
    model.fit(x, y)
    result = action_feats[['feat20_work_year']]
    test_y = model.predict(action_feats.values)
    result['y'] = test_y
    result.reset_index(inplace=True)
    result.sort_values(['user_id', 'y'], ascending=False, inplace=True)
    del result['y']
    result.to_csv(data_path + 'result.csv')


