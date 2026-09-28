# %%
import pandas as pd

from sklearn import model_selection
from sklearn import tree
from sklearn import pipeline
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import ensemble
from sklearn import metrics

import mlflow

from feature_engine import discretisation, encoding

import matplotlib.pyplot as plt

mlflow.set_tracking_uri('http://127.0.0.1:5000/')

mlflow.set_experiment(experiment_name='Churn_experiment')

df = pd.read_csv('../data/archive/abt_churn.csv')
df.head()
# %%
oot = df[df['dtRef'] == df['dtRef'].max()].copy()
oot

# %%
df_train = df[df['dtRef'] < df['dtRef'].max()].copy()
df_train.shape
# %%
df_train.head(5)
features = df_train.columns[2:-1]
target = 'flagChurn'

X = df_train[features]
y = df_train[target]

# %%
# SAMPLE

# %%

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
# %%
print('Taxa variável resposta treino', y_train.mean())
print('Taxa variável resposta teste', y_test.mean())
# %%

# EXPLORE
# %%
X_train.isna().sum()
# %%
df_analise = X_train.copy()
df_analise[target] = y_train
sumario = df_analise.groupby(by=target).agg({'mean','median'}).T
sumario['diff_abs'] = sumario[0] - sumario[1]
sumario['diff_rel'] = sumario[0] / sumario[1]

sumario.sort_values(by=['diff_rel'], ascending=False)
# %%

arvore = tree.DecisionTreeClassifier(random_state=42)

arvore.fit(X_train, y_train)
# %%

plt.figure(dpi=400)
tree.plot_tree(arvore, feature_names=X_train.columns, filled=True, class_names=arvore.classes_.astype(str))
# %%
feature_importance = pd.Series(arvore.feature_importances_, index=X_train.columns).sort_values(ascending=False).reset_index()

feature_importance['acc'] = feature_importance[0].cumsum()
feature_importance[feature_importance['acc'] < 0.96]
# %%
# MODIFY

# %%
best_features = feature_importance[feature_importance['acc'] < 0.96]['index'].to_list()
# %%

tree_discretisation = discretisation.DecisionTreeDiscretiser(variables=best_features, regression=False, cv=3, bin_output='bin_number')

#One Hot Encoding
onehot = encoding.OneHotEncoder(variables=best_features, ignore_format=True)

# %%
# MODEL

# reg = linear_model.LogisticRegression(penalty=None, random_state=42, max_iter=1000000)
# model = naive_bayes.BernoulliNB()
model = ensemble.RandomForestClassifier(random_state=42, n_jobs=2)

params = {
    'min_samples_leaf':[15, 20, 25, 30, 50],
    'n_estimators': [100, 200, 500, 1000],
    'criterion': ['gini', 'entropy', 'log_loss']
}

grid = model_selection.GridSearchCV(model, params, cv=3, scoring='roc_auc', verbose=3)

model_pipeline = pipeline.Pipeline(steps=[('Discretizar', tree_discretisation), ('OneHot', onehot), ('Grid', grid)])

# %%
with mlflow.start_run():
    mlflow.sklearn.autolog()
    model_pipeline.fit(X_train[best_features], y_train)
    y_train_predict = model_pipeline.predict(X_train[best_features])
    y_train_predict_proba = model_pipeline.predict_proba(X_train[best_features])[:,1]

    #ASSESS

    acc_train = metrics.accuracy_score(y_train, y_train_predict)
    auc_train = metrics.roc_auc_score(y_train, y_train_predict_proba)
    roc_train = metrics.roc_curve(y_train, y_train_predict_proba)
    print('Acurácia Treino: ', acc_train)
    print('AUC Treino: ', auc_train)
    y_test_predict = model_pipeline.predict(X_test[best_features])
    y_test_predict_proba = model_pipeline.predict_proba(X_test[best_features])[:,1]

    acc_test = metrics.accuracy_score(y_test, y_test_predict)
    auc_test = metrics.roc_auc_score(y_test, y_test_predict_proba)
    roc_test = metrics.roc_curve(y_train, y_train_predict_proba)
    print('Acurácia Treino: ', acc_test)
    print('AUC Treino: ', auc_test)
    y_oot_predict = model_pipeline.predict(oot[best_features])
    y_oot_predict_proba = model_pipeline.predict_proba(oot[best_features])[:,1]

    acc_oot = metrics.accuracy_score(oot[target], y_oot_predict)
    auc_oot = metrics.roc_auc_score(oot[target], y_oot_predict_proba)
    roc_oot = metrics.roc_curve(oot[target], y_oot_predict_proba)
    print('Acurácia Treino: ', acc_oot)
    print('AUC Treino: ', auc_oot)

    mlflow.log_metrics(
        {
            'acc_train': acc_train,
            'auc_train': auc_train,
            'acc_test': acc_test,
            'auc_test': auc_test,
            'acc_oot': acc_oot,
            'auc_oot': auc_oot
        }
    )
# %%
plt.figure(dpi=400)

plt.plot(roc_train[0], roc_train[1])
plt.plot(roc_test[0], roc_test[1])
plt.plot(roc_oot[0], roc_oot[1])
plt.grid(True)
plt.title('Curva ROC Treino, Teste e Out Of Time')
plt.legend([
    f'Treino: {100*auc_train:.2f}',
    f'Teste: {100*auc_test:.2f}',
    f'Out Of Time: {100*auc_oot:.2f}'
])
# %%
model_df = pd.Series({'model': model_pipeline,
           'features': best_features
})

model_df.to_pickle('model.pkl')
# %%
