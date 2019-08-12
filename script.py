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
from sklearn.model_selection import GridSearchCV
import os
import jieba
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
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


def split_str(x, pattern = [',', ' ', '，', ')', '）', '(', '（', '|'], sep='/'):
    if pd.isna(x):
        return x
    else:
        for p in pattern:
            x = x.replace(p, sep)
        return set(x.split(sep))


def del_key(x, keys):
    if pd.isna(x):
        return x
    for k in keys:
        x = x.replace(k, '')
    return x


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
    keys = ['自我评价|', '操作|', '实习|', '申请|', '资金|', '政策|', '知识|', '大型|', '发布|', '变更|', '传达|', '发光|', '方法论|']
    user['experience_drop_keys'] = user['experience'].apply(del_key, **{'keys': keys})
    user['experience_set'] = user['experience_drop_keys'].apply(split_str, **{'pattern': [], 'sep': '|'})
    return user


def clean_job(raw_job):
    job = raw_job.copy()
    job['jd_title_list'] = job['jd_title'].apply(jieba.lcut)
    job['jd_title_set'] = job['jd_title_list'].apply(lambda x: set(x))
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
    return raw_action.set_index(['user_id', 'jd_no'])


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
    action_feats['feat_live_city_is_big'] = action['live_city_id'].apply(lambda x: big_city_dict[x])
    action_feats['feat_live_city_desire_0'] = (action['live_city_id'] == action['desire_jd_city_id_0'])*1
    action_feats['feat_live_city_desire_1'] = (action['live_city_id'] == action['desire_jd_city_id_1'])*1
    action_feats['feat_live_city_desire_2'] = (action['live_city_id'] == action['desire_jd_city_id_2'])*1
    action_feats['feat_live_ctiy_desire'] = ((action_feats['feat_live_city_desire_0'] + \
                                             action_feats['feat_live_city_desire_1'] + \
                                             action_feats['feat_live_city_desire_2']) >= 1)*1

    action_feats['feat_city_live_job'] = (action['live_city_id'] == action['city'])*1
    action_feats['feat_job_city_desire_0'] = (action['city'] == action['desire_jd_city_id_0'])*1
    action_feats['feat_job_city_desire_1'] = (action['city'] == action['desire_jd_city_id_1'])*1
    action_feats['feat_job_city_desire_2'] = (action['city'] == action['desire_jd_city_id_2'])*1
    action_feats['feat_job_ctiy_desire'] = ((action_feats['feat_job_city_desire_0'] + \
                                             action_feats['feat_job_city_desire_1'] + \
                                             action_feats['feat_job_city_desire_2']) >= 1)*1
    action_feats['feat_desire_cur_indu_len'] = (action['desire_jd_industry_set']-(action['desire_jd_industry_set']-action['cur_industry_set'])).apply(find_len_set)
    action_feats['feat_cur_desire_indu_ratio'] = action_feats['feat_desire_cur_indu_len']/(action['desire_jd_industry_set'].apply(find_len_set) + 0.0)
    action_feats['feat_desire_cur_type_len'] = (action['desire_jd_type_set']-(action['desire_jd_type_set']-action['cur_jd_type_set'])).apply(find_len_set)
    action_feats['feat_cur_desire_type_len_ratio'] = action_feats['feat_desire_cur_type_len']/(action['desire_jd_type_set'].apply(find_len_set) + 0.0)
    action_feats['feat_desire_job_type_len'] = (action['desire_jd_type_set']-(action['desire_jd_type_set']-action['jd_sub_type_set'])).apply(find_len_set)
    action_feats['feat_desire_indu_job_key_len'] = (action['desire_jd_industry_set']-(action['desire_jd_industry_set']-action['key_set'])).apply(find_len_set)
    action_feats['feat_desire_type_job_key_len'] = (action['desire_jd_type_set']-(action['desire_jd_type_set']-action['key_set'])).apply(find_len_set)
    action_feats['feat_desire_indu_title_len'] = (action['desire_jd_industry_set']-(action['desire_jd_industry_set']-action['jd_sub_type_set'])).apply(find_len_set)
    action_feats['feat_desire_indu_title_ratio'] = action_feats['feat_desire_indu_title_len']/(action['desire_jd_industry_set'].apply(find_len_set) + 0.0)
    action_feats['feat_cur_indu_title_len'] = (action['cur_industry_set']-(action['cur_industry_set']-action['jd_sub_type_set'])).apply(find_len_set)
    action_feats['feat_cur_indu_title_ratio'] = action_feats['feat_cur_indu_title_len']/(action['cur_industry_set'].apply(find_len_set) + 0.0)
    action_feats['feat_desire_type_title_len'] = (action['desire_jd_type_set']-(action['desire_jd_type_set']-action['jd_sub_type_set'])).apply(find_len_set)
    action_feats['feat_desire_type_title_ratio'] = action_feats['feat_desire_type_title_len']/(action['desire_jd_type_set'].apply(find_len_set) + 0.0)
    action_feats['feat_cur_type_title_len'] = (action['cur_jd_type_set']-(action['cur_jd_type_set']-action['jd_sub_type_set'])).apply(find_len_set)
    action_feats['feat_cur_type_title_ratio'] = action_feats['feat_cur_type_title_len']/(action['cur_jd_type_set'].apply(find_len_set) + 0.0)

    action_feats['feat_desire_cur_salary'] = action['desire_jd_salary_id'] - action['cur_salary_id']
    action_feats['feat_desire_job_max_salary'] = action['desire_jd_max_salary'] - action['max_salary']
    action_feats['feat_desire_job_min_salary'] = action['desire_jd_min_salary'] - action['min_salary']
    action_feats['feat_birthday'] = action['birthday']
    action_feats['feat_work_year'] = 2019 - action['start_work_date']
    action_feats['feat_start_date_duration'] = (datetime(2019, 9, 1) - action['start_date']).apply(lambda x: x.days)
    action_feats['feat_end_date_duration'] = (datetime(2019, 9, 1) - action['end_date']).apply(lambda x: x.days)
    action_feats['feat_end_start_date_duration'] = (action['end_date'] - action['start_date']).apply(lambda x: x.days)
    action_feats['feat_degree_min_edu'] = action['cur_degree_id'] - action['min_edu_level']
    action_feats['feat_degree'] = action['cur_degree_id']
    action_feats['feat_desire_jd_city_count'] = (action['desire_jd_city_id_0'] != '-')*1 +  \
                                           (action['desire_jd_city_id_1'] != '-')*1 + \
                                           (action['desire_jd_city_id_2'] != '-')*1
    action_feats['feat_desire_jd_industry_count'] = action['desire_jd_industry_set'].apply(find_len_set)
    action_feats['feat_desire_jd_type_count'] = action['desire_jd_type_set'].apply(find_len_set)
    action_feats['feat_cur_salary_id'] = action['cur_salary_id']
    action_feats['feat_cur_industry_count'] = action['cur_industry_set'].apply(find_len_set)
    action_feats['feat_cur_jd_type_count'] = action['cur_jd_type_set'].apply(find_len_set)
    action_feats['feat_jd_sub_type_count'] = action['jd_sub_type'].apply(find_len_set)
    action_feats['feat_require_nums'] = action['require_nums']
    return action_feats


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


def cv_test(model, other_paras, cv_name, cv_paras, data, kfold=4):
    '''
    :param model:
    :param cv_paras:
    :param score:
    :param data: DataFrame(['y','delivered','satisfied'])
    :param kfold:
    :return:
    '''
    data_count = data.shape[0]
    fold_count = data_count / kfold
    fold_dict, test_data_list, train_data_list, score = {}, [], [], {}
    for i in range(kfold):
        fold_dict[i] = [i * fold_count, (i+1) * fold_count]
    for i in range(kfold):
        test_data = data.iloc[fold_dict[i][0]:fold_dict[i][1], :].copy()
        train_data = data.drop(index=test_data.index).copy()
        test_data_list.append(test_data)
        train_data_list.append(train_data)
    for para in cv_paras:
        score[para] = []
        for i in range(kfold):
            print 'para %f , %dth fold start' % (para, i)
            train_data = train_data_list[i]
            test_data = test_data_list[i]
            other_paras[cv_name] = para
            pred_model = model(**other_paras)
            pred_model.fit(train_data.drop(columns=['y', 'delivered', 'satisfied']).values, train_data['y'].values)
            y = pred_model.predict(test_data.drop(columns=['y', 'delivered', 'satisfied']).values)
            test_true = test_data[['delivered', 'satisfied']].reset_index()
            test_true.columns = ['user_id', 'job_id', 'delivered', 'satisfied']
            test_pred = test_data[[]].reset_index()
            test_pred['score'] = y
            test_pred.columns = ['user_id', 'job_id', 'score']
            score[para].append(my_score(test_true, test_pred))
        print '%s is %f, mean score is %f' % (cv_name, para, np.mean(score['para']))
        print para
        print np.mean(score['para'])
    return score


def show_cv_result(gscv):
    print('参数的最佳取值：{0}'.format(gscv.best_params_))
    print('最佳模型得分:{0}'.format(gscv.best_score_))


def set_count(df):
    df_dict = defaultdict(lambda: 0)
    for df_set in df:
        if pd.notna(df_set):
            for content in df_set:
                df_dict[content] = df_dict[content] + 1
    df_pd = pd.DataFrame({'id': df_dict.keys(),  'count': df_dict.values()})
    df_pd.sort_values('count',  ascending=False, inplace=True)
    return set(df_pd['id']), df_pd


def cal_key_grade(df, key_col_name, grade_col_name):
    '''
    :param df: df[key_col_name, grade_col_name]
    :return: df
    '''
    df_dict = defaultdict(lambda: 0)
    for i in range(df.shape[0]):
        if pd.notna(df.iloc[i][key_col_name]):
            for content in df.iloc[i][key_col_name]:
                df_dict[content] = df_dict[content] + df.iloc[i][grade_col_name]
    df_pd = pd.DataFrame({'key': df_dict.keys(),  'grade': df_dict.values()})
    df_pd.sort_values('grade',  ascending=False, inplace=True)
    return df_pd


if __name__ == "__main__":
    set_format()
    data_path = '/Users/botaofan/PycharmProjects/tianchi/zhilian/data/'
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
    test_user_dtype = {'live_city_id': object, 'desire_jd_salary_id': object, 'cur_salary_id': object, 'birthday': np.int16,
                       'start_work_date': object}
    test_user = pd.read_csv(data_path + "user_ToBePredicted", delimiter="\t", error_bad_lines=False, dtype=test_user_dtype)
    #load test_action
    test_action = pd.read_csv(data_path + "zhaopin_round1_user_exposure_A_20190723", delim_whitespace=True)
    user_action = raw_action.groupby(['user_id', 'jd_no'])['delivered', 'satisfied'].max().reset_index()
    #======Data Preprocess======
    user = clean_user(raw_user)
    job = clean_job(raw_job)
    action = train_action_generate(raw_action)
    action_feats = feats_generate(action, user, job)

    pred_user = clean_user(test_user)
    pred_job = job.copy()
    pred_action = test_action_generate(test_action)
    pred_action_feats = feats_generate(pred_action, pred_user, pred_job)




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

    #============Tuning xgboost============
    train_ratio = 0.8
    action_feats_num = action_feats.shape[0]
    train_num = int(train_ratio * action_feats_num)
    train_x = action_feats.iloc[:train_num, :].drop(columns=['y', 'delivered', 'satisfied']).values
    train_y = action_feats.iloc[:train_num, :][['y']].values
    test_x = action_feats.iloc[train_num:, :].drop(columns=['y', 'delivered', 'satisfied']).values
    test_true = action_feats.iloc[train_num:, :][['delivered', 'satisfied']].reset_index()
    test_true.columns = ['user_id', 'job_id', 'delivered', 'satisfied']
    test_pred = action_feats.iloc[train_num:, :][[]].reset_index()
    test_pred.columns = ['user_id', 'job_id']
    xgb_test_score = {}


    #Get MAP score without any tuning
    xgb_model = xgb.XGBRegressor()
    xgb_model.fit(train_x, train_y)
    y_pred = xgb_model.predict(test_x)
    '''
    paras:
    (base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
       max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
       n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=1)
    '''
    test_pred['score'] = y_pred
    xgb_test_score_no_tuning = my_score(test_true, test_pred)#MAP:0.21886006721219667
    #Tuning n_estimators
    params = {'subsample': 1,'colsample_bytree': 1, 'gamma': 0, 'learning_rate': 0.1, 'max_depth': 3,
              'min_child_weight': 1, 'n_estimators': 100, 'n_jobs': 4, 'random_state': 0, 'reg_alpha': 0, 'reg_lambda': 1,
              'scale_pos_weight': 1, 'silent': True}
    cv_params = {'n_estimators': [50, 75, 100, 150, 200]}
    xgb_model = xgb.XGBRegressor(**params)
    gscv = GridSearchCV(estimator=xgb_model, param_grid=cv_params, scoring='neg_mean_squared_error', cv=4,
                        verbose=1, n_jobs=4)
    gscv.fit(train_x, train_y)
    show_cv_result(gscv)
    y_pred = gscv.predict(test_x)
    test_pred['score'] = y_pred
    xgb_test_score['n_estimator_100'] = my_score(test_true, test_pred) # n_estimator = 100, MAP: 0.21886006721219667

    #Tuning min_child_weight and max_depth
    params = {'subsample': 1,'colsample_bytree': 1, 'gamma': 0, 'learning_rate': 0.1, 'max_depth': 3,
              'min_child_weight': 1, 'n_estimators': 100, 'n_jobs': 4, 'random_state': 0, 'reg_alpha': 0, 'reg_lambda': 1,
              'scale_pos_weight': 1, 'silent': True}
    cv_params = {'max_depth': [2, 3, 4, 6], 'min_child_weight': [1, 2, 3, 4]}
    xgb_model = xgb.XGBRegressor(**params)
    gscv = GridSearchCV(estimator=xgb_model, param_grid=cv_params, scoring='neg_mean_squared_error', cv=4,
                        verbose=1, n_jobs=4)
    gscv.fit(train_x, train_y)
    show_cv_result(gscv)
    y_pred = gscv.predict(test_x)
    test_pred['score'] = y_pred
    xgb_test_score['max_depth_4_min_child_weight_4'] = my_score(test_true, test_pred)#max_depth:4, min_child_weight:4 , MAP:0.22011659413590243

    #Tuning gamma
    params = {'subsample': 1, 'colsample_bytree': 1, 'gamma': 0, 'learning_rate': 0.1, 'max_depth': 4,
              'min_child_weight': 4, 'n_estimators': 100, 'n_jobs': 4, 'random_state': 0, 'reg_alpha': 0, 'reg_lambda': 1,
              'scale_pos_weight': 1, 'silent': True}
    cv_params = {'gamma': [0.1, 0.3, 0.5, 0.7]}
    xgb_model = xgb.XGBRegressor(**params)
    gscv = GridSearchCV(estimator=xgb_model, param_grid=cv_params, scoring='neg_mean_squared_error', cv=4,
                        verbose=1, n_jobs=4)
    gscv.fit(train_x, train_y)
    show_cv_result(gscv)
    y_pred = gscv.predict(test_x)
    test_pred['score'] = y_pred
    xgb_test_score['gamma_0.7'] = my_score(test_true, test_pred)#gamma:0.7, MAP:0.2228540359573349

    #Tuning subsample and colsample_bytree:
    params = {'subsample': 1, 'colsample_bytree': 1, 'gamma': 0.7, 'learning_rate': 0.1, 'max_depth': 4,
              'min_child_weight': 4, 'n_estimators': 100, 'n_jobs': 4, 'random_state': 0, 'reg_alpha': 0, 'reg_lambda': 1,
              'scale_pos_weight': 1, 'silent': True,}
    cv_params = {'subsample': [0.6, 0.7, 0.8, 0.9], 'colsample_bytree': [0.6, 0.7, 0.8, 0.9]}
    xgb_model = xgb.XGBRegressor(**params)
    gscv = GridSearchCV(estimator=xgb_model, param_grid=cv_params, scoring='neg_mean_squared_error', cv=4,
                        verbose=1, n_jobs=4)
    gscv.fit(train_x, train_y)
    show_cv_result(gscv)
    y_pred = gscv.predict(test_x)
    test_pred['score'] = y_pred
    xgb_test_score['subsample_0.7_colsample_bytree_0.6'] = my_score(test_true, test_pred)#subsample:0.7_colsample_bytree:0.6, MAP:0.2228540359573349
    # Tuning subsample and colsample_bytree:
    params = {'subsample': 1, 'colsample_bytree': 1, 'gamma': 0.7, 'learning_rate': 0.1, 'max_depth': 4,
              'min_child_weight': 4, 'n_estimators': 100, 'n_jobs': 4, 'random_state': 0, 'reg_alpha': 0, 'reg_lambda': 1,
              'scale_pos_weight': 1, 'silent': True}
    cv_params = {'reg_alpha': [0, 0.1, 1, 2], 'reg_lambda': [0, 0.1, 1, 2]}
    xgb_model = xgb.XGBRegressor(**params)
    gscv = GridSearchCV(estimator=xgb_model, param_grid=cv_params, scoring='neg_mean_squared_error', cv=4,
                        verbose=1, n_jobs=4)
    gscv.fit(train_x, train_y)
    show_cv_result(gscv)
    y_pred = gscv.predict(test_x)
    test_pred['score'] = y_pred
    xgb_test_score['reg_alpha_2_reg_lambda_0.1'] = my_score(test_true, test_pred)#reg_alpha:2, reg_lambda:0, MAP:0.2228540359573349


    #Training
    params = {'subsample': 1, 'colsample_bytree': 1, 'gamma': 0.7, 'learning_rate': 0.1, 'max_depth': 4,
              'min_child_weight': 4, 'n_estimators': 100, 'n_jobs': 4, 'random_state': 0, 'reg_alpha': 0, 'reg_lambda': 1,
              'scale_pos_weight': 1, 'silent': True}
    xgb_model = xgb.XGBRegressor(**params)

    train_x = action_feats.iloc[:400000, :].drop(columns=['y', 'delivered', 'satisfied']).values
    train_y = action_feats.iloc[:400000, :][['y']].values
    xgb_model.fit(train_x, train_y)
    train_true = action_feats.iloc[:400000,:][['delivered', 'satisfied']].reset_index()
    train_true.columns = ['user_id', 'job_id', 'delivered', 'satisfied']
    train_pred = action_feats.iloc[:400000, :][[]].reset_index()
    train_pred['score'] = xgb_model.predict(train_x)
    train_pred.columns = ['user_id', 'job_id', 'score']
    print 'Score on Train set(about 400000 records) is %f' % my_score(train_true, train_pred)

    test_x = action_feats.iloc[400000:, :].drop(columns=['y', 'delivered', 'satisfied']).values
    test_true = action_feats.iloc[400000:, :][['delivered', 'satisfied']].reset_index()
    test_true.columns = ['user_id', 'job_id', 'delivered', 'satisfied']
    test_pred = action_feats.iloc[400000:, :][[]].reset_index()
    test_pred['score'] = xgb_model.predict(test_x)
    test_pred.columns = ['user_id', 'job_id', 'score']
    print 'Score on Test set(about 400000 records) is %f' % my_score(test_true, test_pred)



    pred_result = xgb_model.predict(pred_action_feats.values)
    result = pred_action_feats[[]].reset_index()
    result['score'] = pred_result
    result.sort_values(['user_id', 'score'], ascending=False, inplace=True)



    #Tuning lightboost
    lgb_params = {'boosting_type': 'gbdt',
                'objective': 'regression',
                'num_boost_round':100,
                'metrics':'mse',
                'learning_rate': 0.1,
                'num_leaves': 50,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8
                }
    cv_params = {'num_boost_round': [25, 50, 75, 100, 125]}
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    gscv = GridSearchCV(estimator=lgb_model, param_grid=cv_params, scoring='neg_mean_squared_error', cv=5,
                            verbose=1, n_jobs=4)
    gscv.fit(x, y)




