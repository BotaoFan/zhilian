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
lgb_params = {'boosting_type': 'gbdt',
              'objective': 'regression',
              'num_boost_round': 100,
              'metrics': 'mse',
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
