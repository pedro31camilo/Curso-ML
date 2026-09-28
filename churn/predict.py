# %%
import pandas as pd
import mlflow.sklearn
import mlflow

mlflow.set_tracking_uri('http://127.0.0.1:5000/')

models = mlflow.search_registered_models(filter_string="name = 'Churn_model'")
latest_version = max([i.version for i in models[0].latest_versions])
latest_version

model = mlflow.sklearn.load_model(f'models:/Churn_model/{latest_version}')
features = model.feature_names_in_
features
# %%
df = pd.read_csv('../data/archive/abt_churn.csv')
amostra = df[df['dtRef'] == df['dtRef'].max()].sample(3)
amostra = amostra.drop('flagChurn', axis=1)
# %%
predicao = model.predict_proba(amostra[features])[:, 1]
amostra['proba'] = predicao
amostra
