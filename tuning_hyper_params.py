#-*- coding:utf-8 -*-
# @Time : 2019/8/13
# @Author : Botao Fan

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

# ============Tuning LightGBM============
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

lgb_params = {'boosting_type': 'gbdt', 'objective': 'regression', 'metrics': 'mse',
              'num_boost_round': 150, 'learning_rate': 0.03, 'num_leaves': 128, 'max_depth': 9,
              'min_child_samples': 10, 'min_child_weight': 0.001, 'reg_alpha': 0, 'reg_lambda': 3,
              'feature_fraction': 0.6, 'bagging_fraction': 0.9, 'bagging_freq': 5}