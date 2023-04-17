import pandas as pd

from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.under_sampling import RandomUnderSampler

from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import FactorAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


train_df = pd.read_csv('./dataset.csv')
X = train_df.drop('target', axis=1)
y = train_df['target']


# =====================================================================================================
# # CatBoost:
#
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_auc_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.linear_model import LogisticRegression
#
# from catboost import CatBoostClassifier
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=7, stratify=y)
#
# scaler = StandardScaler()
# pca = PCA(n_components=3, random_state=42)
#
# X_train_pca = pca.fit_transform(scaler.fit_transform(X_train))
# X_train_with_val, X_val, y_train_with_val, y_val = \
#     train_test_split(X_train_pca, y_train_pca, test_size=0.1, random_state=42, stratify=y_train_pca)
#
# model = CatBoostClassifier(eval_metric='AUC:hints=skip_train~false', logging_level='Silent',
#                            early_stopping_rounds=150, random_state=42, auto_class_weights='Balanced')
# model.fit(X_train_with_val, y_train_with_val, eval_set=(X_val, y_val))
#
# print(roc_auc_score(y_test, model.predict(pca.transform(scaler.transform(X_test))))) 0.692
#
#
# =====================================================================================================
# # GridSearch:
#
# from imblearn.ensemble import EasyEnsembleClassifier
# from imblearn.ensemble import BalancedRandomForestClassifier
# from imblearn.ensemble import BalancedBaggingClassifier
# from imblearn.ensemble import RUSBoostClassifier
# from imblearn.over_sampling import SMOTE
#
# from sklearn.decomposition import FactorAnalysis
# from sklearn.decomposition import PCA
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.model_selection import GridSearchCV
# from sklearn.pipeline import Pipeline
# from sklearn.tree import DecisionTreeClassifier
#
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=7, stratify=y)
#
# scaler = StandardScaler()
# # pca = PCA(n_components=3, max_iter=10000)
# fa = FactorAnalysis(n_components=3, random_state=42)
#
# logloss_estimator = DecisionTreeClassifier(criterion='log_loss', random_state=42)
# entropy_estimator = DecisionTreeClassifier(criterion='entropy', random_state=42)
# gini_estimator = DecisionTreeClassifier(criterion='gini', random_state=42)
#
# sampler = RandomUnderSampler(random_state=42)
#
# model = BalancedBaggingClassifier(estimator, oob_score=True, sampler=sampler, random_state=42)
#
# pipeline = Pipeline(steps=[('scaler', scaler), ('fa', fa), ('model', model)])
#
# params = {
#     'fa__n_components: [2, 3, 4, 5, 10]',
#     'fa__max_iter': [100, 500, 1000],
#     'model__estimator': [logloss_estimator, entropy_estimator, gini_estimator],
#     'model__n_estimators': [100, 500, 1000],
#     'model__oob_score': [True, False],
# }
#
# search = GridSearchCV(pipeline, params, n_jobs=-1)
# search.fit(X_train, y_train)
#
# print(search.best_params_)
# print(roc_auc_score(y_test, search.predict(X_test))) # 0.752
#
#
# =====================================================================================================
# # Cross Validation
#
# from imblearn.ensemble import EasyEnsembleClassifier
# from imblearn.ensemble import BalancedRandomForestClassifier
# from imblearn.ensemble import BalancedBaggingClassifier
# from imblearn.ensemble import RUSBoostClassifier
# from imblearn.over_sampling import SMOTE
#
# from sklearn.decomposition import FactorAnalysis
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.model_selection import StratifiedKFold
# from sklearn.ensemble import BaggingClassifier
#
#
# ans = 0
# N = 10
#
# for _ in range(N):
#     avg_score = 0
#     num_splits = 10
#
#     for (train_index, test_index) in StratifiedKFold(num_splits).split(X, y):
#         scaler = StandardScaler()
#         fa = FactorAnalysis(n_components=3, max_iter=1000)
#
#         X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y[train_index], y[test_index]
#         X_train_fa = fa.fit_transform(scaler.fit_transform(X_train))
#
#         estimator = DecisionTreeClassifier(criterion='log_loss')
#         model = BalancedBaggingClassifier(estimator, n_estimators=500, oob_score=False)
#         model.fit(X_train_fa, y_train)
#
#         avg_score += roc_auc_score(y_test, model.predict(fa.transform(scaler.transform(X_test))))
#
#     ans += avg_score / num_splits
#     print(avg_score / num_splits)
#
# print(f'Total Average: {ans / N}') # 0.704
#


scaler = StandardScaler()
fa = FactorAnalysis(n_components=3, max_iter=1000, random_state=42)
estimator = DecisionTreeClassifier(criterion='log_loss', random_state=42)
sampler = RandomUnderSampler(random_state=42)
model = BalancedBaggingClassifier(estimator, n_estimators=500, sampler=sampler, oob_score=False, random_state=42)

pipeline = Pipeline(steps=[('scaler', scaler), ('fa', fa), ('model', model)])
pipeline.fit(X, y)

test_df = pd.read_csv('./test.csv', index_col='id')

prediction = pd.DataFrame(pipeline.predict(test_df), index=test_df.index, columns=['target'])
prediction.to_csv('./submission.csv')
