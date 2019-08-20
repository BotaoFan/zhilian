#-*- coding:utf-8 -*-
# @Time : 2019/8/13
# @Author : Botao Fan
from time import time
# ============Tuning XGBoost============
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

# Get MAP score without any tuning
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
xgb_test_score_no_tuning = my_score(test_true, test_pred)  # MAP:0.21886006721219667
# Tuning n_estimators
params = {'subsample': 1, 'colsample_bytree': 1, 'gamma': 0, 'learning_rate': 0.1, 'max_depth': 3,
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
xgb_test_score['n_estimator_100'] = my_score(test_true, test_pred)  # n_estimator = 100, MAP: 0.21886006721219667

# Tuning min_child_weight and max_depth
params = {'subsample': 1, 'colsample_bytree': 1, 'gamma': 0, 'learning_rate': 0.1, 'max_depth': 3,
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
xgb_test_score['max_depth_4_min_child_weight_4'] = my_score(test_true,
                                                            test_pred)  # max_depth:4, min_child_weight:4 , MAP:0.22011659413590243

# Tuning gamma
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
xgb_test_score['gamma_0.7'] = my_score(test_true, test_pred)  # gamma:0.7, MAP:0.2228540359573349

# Tuning subsample and colsample_bytree:
params = {'subsample': 1, 'colsample_bytree': 1, 'gamma': 0.7, 'learning_rate': 0.1, 'max_depth': 4,
          'min_child_weight': 4, 'n_estimators': 100, 'n_jobs': 4, 'random_state': 0, 'reg_alpha': 0, 'reg_lambda': 1,
          'scale_pos_weight': 1, 'silent': True, }
cv_params = {'subsample': [0.6, 0.7, 0.8, 0.9], 'colsample_bytree': [0.6, 0.7, 0.8, 0.9]}
xgb_model = xgb.XGBRegressor(**params)
gscv = GridSearchCV(estimator=xgb_model, param_grid=cv_params, scoring='neg_mean_squared_error', cv=4,
                    verbose=1, n_jobs=4)
gscv.fit(train_x, train_y)
show_cv_result(gscv)
y_pred = gscv.predict(test_x)
test_pred['score'] = y_pred
xgb_test_score['subsample_0.7_colsample_bytree_0.6'] = my_score(test_true,
                                                                test_pred)  # subsample:0.7_colsample_bytree:0.6, MAP:0.2228540359573349
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
xgb_test_score['reg_alpha_2_reg_lambda_0.1'] = my_score(test_true,
                                                        test_pred)  # reg_alpha:2, reg_lambda:0, MAP:0.2228540359573349

# ============Tuning LightGBM（The way doesn't work well）============
lgb_test_score = {}
# Tuning num_boost_round
lgb_params = {'boosting_type': 'gbdt', 'objective': 'regression', 'metrics': 'mse',
              'num_boost_round': 100, 'learning_rate': 0.1, 'num_leaves': 50, 'max_depth': 6, 'subsample': 0.8,
              'colsample_bytree': 0.8}
cv_params = {'num_boost_round': [25, 50, 75, 100, 125]}
lgb_model = lgb.LGBMRegressor(**lgb_params)
gscv = GridSearchCV(estimator=lgb_model, param_grid=cv_params, scoring='neg_mean_squared_error', cv=5,
                    verbose=1, n_jobs=4)
gscv.fit(train_x, train_y)
show_cv_result(gscv)
y_pred = gscv.predict(test_x)
test_pred['score'] = y_pred
lgb_test_score['num_boost_round_50'] = my_score(test_true, test_pred)  # gamma:0.7, MAP:0.22295576013341234

# Tuning max_depth and num_leaves
lgb_params = {'boosting_type': 'gbdt', 'objective': 'regression', 'metrics': 'mse',
              'num_boost_round': 50, 'learning_rate': 0.1, 'num_leaves': 50, 'max_depth': 6, 'subsample': 0.8,
              'colsample_bytree': 0.8}
cv_params = {'max_depth': [3, 5, 7, 9], 'num_leaves': [8, 32, 128, 512]}
lgb_model = lgb.LGBMRegressor(**lgb_params)
gscv = GridSearchCV(estimator=lgb_model, param_grid=cv_params, scoring='neg_mean_squared_error', cv=5,
                    verbose=1, n_jobs=4)
gscv.fit(train_x, train_y)
show_cv_result(gscv)
y_pred = gscv.predict(test_x)
test_pred['score'] = y_pred
lgb_test_score['max_depth_7_num_leaves_32'] = my_score(test_true, test_pred)  # MAP:0.22345826302411678

# Tuning min_child_samples and min_child_weight
lgb_params = {'boosting_type': 'gbdt', 'objective': 'regression', 'metrics': 'mse',
              'num_boost_round': 50, 'learning_rate': 0.1, 'num_leaves': 32, 'max_depth': 7, 'subsample': 0.8,
              'colsample_bytree': 0.8}
cv_params = {'min_child_samples': [16, 18, 20, 22], 'min_child_weight': [0.001, 0.002, 0.003]}
lgb_model = lgb.LGBMRegressor(**lgb_params)
gscv = GridSearchCV(estimator=lgb_model, param_grid=cv_params, scoring='neg_mean_squared_error', cv=5,
                    verbose=1, n_jobs=4)
gscv.fit(train_x, train_y)
show_cv_result(gscv)
y_pred = gscv.predict(test_x)
test_pred['score'] = y_pred
lgb_test_score['min_child_samples_16_min_child_weight_0.001'] = my_score(test_true, test_pred)  #  MAP:0.22152546764910555

# Tuning feature_fraction and bagging_fraction
lgb_params = {'boosting_type': 'gbdt', 'objective': 'regression', 'metrics': 'mse',
              'num_boost_round': 50, 'learning_rate': 0.1, 'num_leaves': 32, 'max_depth': 7,
              'min_child_samples': 16, 'min_child_weight': 0.001,
              'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5}
cv_params = {'feature_fraction': [0.6, 0.7, 0.8, 0.9], 'bagging_fraction': [0.6, 0.7, 0.8, 0.9]}
lgb_model = lgb.LGBMRegressor(**lgb_params)
gscv = GridSearchCV(estimator=lgb_model, param_grid=cv_params, scoring='neg_mean_squared_error', cv=5,
                    verbose=1, n_jobs=4)
gscv.fit(train_x, train_y)
show_cv_result(gscv)
y_pred = gscv.predict(test_x)
test_pred['score'] = y_pred
lgb_test_score['feature_fraction_0.6_bagging_fraction_0.9'] = my_score(test_true, test_pred)  #  MAP:0.22348834147483604

# Tuning reg_alpha and reg_lambda
lgb_params = {'boosting_type': 'gbdt', 'objective': 'regression', 'metrics': 'mse',
              'num_boost_round': 50, 'learning_rate': 0.1, 'num_leaves': 32, 'max_depth': 7,
              'min_child_samples': 16, 'min_child_weight': 0.001,
              'feature_fraction': 0.6, 'bagging_fraction': 0.9, 'bagging_freq': 5}
cv_params = {'reg_alpha': [0, 0.01, 0.03, 0.05], 'reg_lambda': [0, 0.1, 0.5, 1, 3]}
lgb_model = lgb.LGBMRegressor(**lgb_params)
gscv = GridSearchCV(estimator=lgb_model, param_grid=cv_params, scoring='neg_mean_squared_error', cv=5,
                    verbose=1, n_jobs=4)
gscv.fit(train_x, train_y)
show_cv_result(gscv)
y_pred = gscv.predict(test_x)
test_pred['score'] = y_pred
lgb_test_score['reg_alpha_0_reg_lambda_3'] = my_score(test_true, test_pred)  #  MAP:0.22348834147483604


#Test
lgb_params = {'boosting_type': 'gbdt', 'objective': 'regression', 'metrics': 'mse',
              'num_boost_round': 50, 'learning_rate': 0.1, 'num_leaves': 32, 'max_depth': 7,
              'min_child_samples': 16, 'min_child_weight': 0.001, 'reg_alpha': 0, 'reg_lambda': 3,
              'feature_fraction': 0.6, 'bagging_fraction': 0.9, 'bagging_freq': 5}
lgb_model = lgb.LGBMRegressor(**lgb_params)
all_count = action_feats.shape[0]
train_count = int(all_count * 0.2)
train_action_feats = action_feats.loc[np.mod(range(all_count),5)!=0, :]
test_action_feats = action_feats.loc[np.mod(range(all_count),5)==0, :]
x = train_action_feats.drop(columns=['y', 'delivered', 'satisfied']).values
y = train_action_feats[['y']].values
lgb_model.fit(x, y)
train_true = train_action_feats[['delivered', 'satisfied']].reset_index()
train_true.columns = ['user_id', 'job_id', 'delivered', 'satisfied']
train_pred = train_action_feats[[]].reset_index()
train_pred['score'] = lgb_model.predict(x)
train_pred.columns = ['user_id', 'job_id', 'score']
print "Train MAP is %f" %(my_score(train_true, train_pred))
test_x = test_action_feats.drop(columns=['y', 'delivered', 'satisfied']).values
test_true = test_action_feats[['delivered', 'satisfied']].reset_index()
test_true.columns = ['user_id', 'job_id', 'delivered', 'satisfied']
test_pred = test_action_feats[[]].reset_index()
test_pred['score'] = lgb_model.predict(test_x)
test_pred.columns = ['user_id', 'job_id', 'score']
print "Train MAP is %f" %(my_score(test_true, test_pred))

# ============Tuning LightGBM============
# -----No.1-----
lgb_params = {'boosting_type': 'gbdt', 'objective': 'regression', 'metrics': 'mse',
              'num_boost_round': 3000, 'learning_rate': 0.002, 'n_estimators': 3000, 'num_leaves': 512, 'max_depth': 9,
              'min_child_samples': 0, 'min_child_weight': 0, 'reg_alpha': 0, 'reg_lambda': 0,
              'feature_fraction': 1, 'bagging_fraction': 1, 'bagging_freq': 5}
lgb_model = lgb.LGBMRegressor(**lgb_params)
lgb_train_score, lgb_test_score = cv_test_model(lgb_model, action_feats, kfold=4)
#train_score:[0.38526964330904245, 0.38728510274477185, 0.38224725572546875, 0.38399374542814474] mean:0.38469893680185696
#test_score:[0.24296956473793827, 0.23270730548362675, 0.23985737772514373, 0.23327056919200723] mean:0.237201204284679
# -----No.2 more overfitting-----
lgb_params = {'boosting_type': 'gbdt', 'objective': 'regression', 'metrics': 'mse',
              'num_boost_round': 4000, 'n_estimators': 4000, 'learning_rate': 0.002, 'num_leaves': 1024, 'max_depth': 10,
              'min_child_samples': 0, 'min_child_weight': 0, 'reg_alpha': 0, 'reg_lambda': 0,
              'feature_fraction': 1, 'bagging_fraction': 1, 'bagging_freq': 5}
lgb_model = lgb.LGBMRegressor(**lgb_params)
t1 = time()
lgb_train_score, lgb_test_score = cv_test_model(lgb_model, action_feats, kfold=4)
t2 = time()
duration = (t2-t1)/60.0
#train_score:[0.42223176081434577, 0.48025631595793894, 0.47707841186507294, 0.46765101720819985] mean:0.4618043764613894
#test_score:[0.24238674537621352, 0.23071295195664915, 0.23900095509362643, 0.22929111428303517] mean:0.23534794167738107
#duration: 34.7min
# -----No.3 more underfitting-----
lgb_params = {'boosting_type': 'gbdt', 'objective': 'regression', 'metrics': 'mse',
              'num_boost_round': 3000, 'n_estimators': 3000, 'learning_rate': 0.002, 'num_leaves': 256, 'max_depth': 8,
              'min_child_samples': 0, 'min_child_weight': 0, 'reg_alpha': 0, 'reg_lambda': 0,
              'feature_fraction': 1, 'bagging_fraction': 1, 'bagging_freq': 5}
lgb_model = lgb.LGBMRegressor(**lgb_params)
t1 = time()
lgb_train_score, lgb_test_score = cv_test_model(lgb_model, action_feats, kfold=4)
t2 = time()
duration = (t2-t1)/60.0
#train_score:[0.333019487928958,0.33446432588425684,0.3288853795677152,0.33466363406072447] mean:0.33275820686041363
#test_score: [0.24076813393011331,0.23442018756163463,0.24022740110528182,0.2338307690616001]mean:0.23731162291465746
#duration: 19.4min
# -----No.4 more underfitting-----
lgb_params = {'boosting_type': 'gbdt', 'objective': 'regression', 'metrics': 'mse',
              'num_boost_round': 3000, 'n_estimators': 3000, 'learning_rate': 0.002, 'num_leaves': 128, 'max_depth': 7,
              'min_child_samples': 0, 'min_child_weight': 0, 'reg_alpha': 0, 'reg_lambda': 0,
              'feature_fraction': 1, 'bagging_fraction': 1, 'bagging_freq': 5}
lgb_model = lgb.LGBMRegressor(**lgb_params)
t1 = time()
lgb_train_score, lgb_test_score = cv_test_model(lgb_model, action_feats, kfold=4)
t2 = time()
duration = (t2-t1)/60.0
#train_score: [0.24223177590544473,0.23177343980749548,0.2373506953279124,0.23526943548837664] mean:0.29475084064547247
#test_score:[0.24223177590544473,0.23177343980749548,0.2373506953279124,0.23526943548837664] mean:0.2366563366323073
#duration: 14.5min

# -----No.5 more underfitting-----
lgb_params = {'boosting_type': 'gbdt', 'objective': 'regression', 'metrics': 'mse',
              'num_boost_round': 3000, 'n_estimators': 3000, 'learning_rate': 0.002, 'num_leaves': 256, 'max_depth': 8,
              'min_child_samples': 10, 'min_child_weight': 0, 'reg_alpha': 0, 'reg_lambda': 0,
              'feature_fraction': 1, 'bagging_fraction': 1, 'bagging_freq': 5}
lgb_model = lgb.LGBMRegressor(**lgb_params)
t1 = time()
lgb_train_score, lgb_test_score = cv_test_model(lgb_model, action_feats, kfold=4)
t2 = time()
duration = (t2-t1)/60.0
#train_score:[0.3228513092527008, 0.32608287664061714, 0.3223114378161002, 0.3220477982636991]  mean:0.32332335549327934
#test_score: [0.2454735203317758, 0.23451597078447883, 0.24159764935337835, 0.23679059269622207] mean:0.23959443329146377
#duration: 17.09min


# -----No.6 more underfitting-----
lgb_params = {'boosting_type': 'gbdt', 'objective': 'regression', 'metrics': 'mse',
              'num_boost_round': 3000, 'n_estimators': 3000, 'learning_rate': 0.002, 'num_leaves': 256, 'max_depth': 8,
              'min_child_samples': 10, 'min_child_weight': 0.002, 'reg_alpha': 0, 'reg_lambda': 0,
              'feature_fraction': 1, 'bagging_fraction': 1, 'bagging_freq': 5}
lgb_model = lgb.LGBMRegressor(**lgb_params)
t1 = time()
lgb_train_score, lgb_test_score = cv_test_model(lgb_model, action_feats, kfold=4)
t2 = time()
duration = (t2-t1)/60.0
#train_score:[0.3228513092527008,0.32608287664061714,0.3223114378161002,0.3220477982636991] mean:0.32332335549327934
#test_score: [0.2454735203317758,0.23451597078447883,0.24159764935337835,0.23679059269622207]mean:0.23959443329146377
#duration: 17.13min


# -----No.7 more underfitting-----
lgb_params = {'boosting_type': 'gbdt', 'objective': 'regression', 'metrics': 'mse',
              'num_boost_round': 3000, 'n_estimators': 3000, 'learning_rate': 0.002, 'num_leaves': 256, 'max_depth': 8,
              'min_child_samples': 10, 'min_child_weight': 0.1, 'reg_alpha': 0, 'reg_lambda': 0,
              'feature_fraction': 1, 'bagging_fraction': 1, 'bagging_freq': 5}
lgb_model = lgb.LGBMRegressor(**lgb_params)
t1 = time()
lgb_train_score, lgb_test_score = cv_test_model(lgb_model, action_feats, kfold=4)
t2 = time()
duration = (t2-t1)/60.0
#train_score:[0.3228513092527008, 0.32608287664061714, 0.3223114378161002, 0.3220477982636991] mean:0.32332335549327934
#test_score: [0.2454735203317758, 0.23451597078447883, 0.24159764935337835, 0.23679059269622207]mean:0.23959443329146377
#duration: 16.88min


# -----No.8 more underfitting-----
lgb_params = {'boosting_type': 'gbdt', 'objective': 'regression', 'metrics': 'mse',
              'num_boost_round': 3000, 'n_estimators': 3000, 'learning_rate': 0.002, 'num_leaves': 256, 'max_depth': 8,
              'min_child_samples': 15, 'min_child_weight': 0.1, 'reg_alpha': 0, 'reg_lambda': 0,
              'feature_fraction': 1, 'bagging_fraction': 1, 'bagging_freq': 5}
lgb_model = lgb.LGBMRegressor(**lgb_params)
t1 = time()
lgb_train_score, lgb_test_score = cv_test_model(lgb_model, action_feats, kfold=4)
t2 = time()
duration = (t2-t1)/60.0
#train_score: [0.24508569088327628, 0.2343704799276639, 0.24017294373545076, 0.23426967996253395]mean:0.3194767601869446
#test_score:[0.24508569088327628, 0.2343704799276639, 0.24017294373545076, 0.23426967996253395] mean:0.2384746986272312
#duration:  17.45min

# -----No.9 more underfitting-----
lgb_params = {'boosting_type': 'gbdt', 'objective': 'regression', 'metrics': 'mse',
              'num_boost_round': 3000, 'n_estimators': 3000, 'learning_rate': 0.002, 'num_leaves': 256, 'max_depth': 8,
              'min_child_samples': 5, 'min_child_weight': 0.1, 'reg_alpha': 0, 'reg_lambda': 0,
              'feature_fraction': 1, 'bagging_fraction': 1, 'bagging_freq': 5}
lgb_model = lgb.LGBMRegressor(**lgb_params)
t1 = time()
lgb_train_score, lgb_test_score = cv_test_model(lgb_model, action_feats, kfold=4)
t2 = time()
duration = (t2-t1)/60.0
#train_score: [0.32929924257062665,0.3331299549502403, 0.32834695288812193, 0.3274253285403742]mean:0.32955036973734075
#test_score: [0.24385238577755503, 0.23340343613085218, 0.24023406810366243, 0.2340234381652093]mean:0.23787833204431974
#duration:  18.35min



# ============Online Test Result Record============
#8-16：lightGBM params and online test map:0.232
lgb_params = {'boosting_type': 'gbdt', 'objective': 'regression', 'metrics': 'mse',
              'num_boost_round': 150, 'learning_rate': 0.03, 'num_leaves': 128, 'max_depth': 9,
              'min_child_samples': 10, 'min_child_weight': 0.001, 'reg_alpha': 0, 'reg_lambda': 3,
              'feature_fraction': 0.6, 'bagging_fraction': 0.9, 'bagging_freq': 5}
lgb_model = lgb.LGBMRegressor(**lgb_params)
lgb_train_score, lgb_test_score = cv_test_model(lgb_model, action_feats, kfold=4)
#8-17：lightGBM params and online test map:0.239
lgb_params = {'boosting_type': 'gbdt', 'objective': 'regression', 'metrics': 'mse',
              'num_boost_round': 3000, 'learning_rate': 0.002, 'n_estimators': 3000, 'num_leaves': 128, 'max_depth': 9,
              'min_child_samples': 8, 'min_child_weight': 0.001, 'reg_alpha': 0, 'reg_lambda': 1,
              'feature_fraction': 0.8, 'bagging_fraction': 0.9, 'bagging_freq': 5}
lgb_model = lgb.LGBMRegressor(**lgb_params)
#8-19:lightGBM params and online test map:
lgb_params = {'boosting_type': 'gbdt', 'objective': 'regression', 'metrics': 'mse',
              'num_boost_round': 6000, 'learning_rate': 0.0015, 'n_estimators': 6000, 'num_leaves': 128, 'max_depth': 9,
              'min_child_samples': 8, 'min_child_weight': 0.001, 'reg_alpha': 0, 'reg_lambda': 1,
              'feature_fraction': 0.8, 'bagging_fraction': 0.9, 'bagging_freq': 5}


#new_feature
lgb_params = {'boosting_type': 'gbdt', 'objective': 'regression', 'metrics': 'mse',
              'num_boost_round': 4000, 'learning_rate': 0.002, 'n_estimators': 4000, 'num_leaves': 2048, 'max_depth': 11,
              'min_child_samples': 4, 'min_child_weight': 0.0005, 'reg_alpha': 3, 'reg_lambda': 3,
              'feature_fraction': 0.6, 'bagging_fraction': 0.6, 'bagging_freq': 5}
'''
[0.4315777576817291, 0.43582763411305, 0.42618078582295943, 0.4281639551725522]
[0.24769832014923102,
 0.24278531881854962,
 0.24886223743913582,
 0.24349999935742234]
 0.2457114689410847'''
lgb_params = {'boosting_type': 'gbdt', 'objective': 'regression', 'metrics': 'mse',
              'num_boost_round': 4000, 'learning_rate': 0.002, 'n_estimators': 4000, 'num_leaves': 2048, 'max_depth': 11,
              'min_child_samples': 6, 'min_child_weight': 0.001, 'reg_alpha': 4, 'reg_lambda': 4,
              'feature_fraction': 0.6, 'bagging_fraction': 0.6, 'bagging_freq': 5}
'''
[0.3996503723018438,
 0.40224449314386984,
 0.3950388814723937,
 0.39843346859477874]
 [0.2482951923285271,
 0.24369054924930345,
 0.24876363297625198,
 0.24368140545545747]
 0.246107695002385
'''
lgb_params = {'boosting_type': 'gbdt', 'objective': 'regression', 'metrics': 'mse',
              'num_boost_round': 4000, 'learning_rate': 0.002, 'n_estimators': 4000, 'num_leaves': 2048, 'max_depth': 11,
              'min_child_samples': 7, 'min_child_weight': 0.001, 'reg_alpha': 5, 'reg_lambda': 5,
              'feature_fraction': 0.5, 'bagging_fraction': 0.5, 'bagging_freq': 5}
'''
[0.36354479678115936,
 0.36717081794495254,
 0.3607854181716067,
 0.3648383345039323]
 [0.24653073084914628,
 0.24097596973451896,
 0.24770538634236824,
 0.24258428656753867]
 0.24444909337339304
'''
lgb_params = {'boosting_type': 'gbdt', 'objective': 'regression', 'metrics': 'mse',
              'num_boost_round': 4500, 'learning_rate': 0.002, 'n_estimators': 4500, 'num_leaves': 2048, 'max_depth': 13,
              'min_child_samples': 7, 'min_child_weight': 0.001, 'reg_alpha': 6, 'reg_lambda': 6,
              'feature_fraction': 0.6, 'bagging_fraction': 0.6, 'bagging_freq': 5}