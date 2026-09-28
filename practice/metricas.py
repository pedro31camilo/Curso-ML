#%%
import pandas as pd

url = "https://docs.google.com/spreadsheets/d/1YQBQ3bu1TCmgrRch1gzW5O4Jgc8huzUSr7VUkxg0KIw/export?gid=283387421&format=csv"
df = pd.read_csv(url)
df.head()
# %%
df['Posição da cadeira (senioridade)'].unique()
df.columns
# %%
df = df.replace({'Sim': 1, 'Não': 0})
df.head()
# %%
num_vars = ['Curte games?', 'Curte futebol?', 'Curte livros?', 'Curte jogos de tabuleiro?', 'Curte jogos de fórmula 1?', 'Curte jogos de MMA?', 'Idade' ]

dummy_vars = [
    'Como conheceu o Téo Me Why?',
    'Quantos cursos acompanhou do Téo Me Why?',
    'Estado que mora atualmente',
    'Área de Formação',
    'Tempo que atua na área de dados',
    'Posição da cadeira (senioridade)'
]

df_analise = pd.get_dummies(df[dummy_vars]).astype(int)

df_analise[num_vars] = df[num_vars].copy()

df_analise.head()

df_analise['pessoa feliz'] = df['Você se considera uma pessoa feliz?'].copy()

# %%
df_analise.columns

# %%
from sklearn import tree
from sklearn import linear_model
from sklearn import naive_bayes

features = list(df_analise.drop(columns=['pessoa feliz']).columns)

X = df_analise.drop(columns=['pessoa feliz'])
y = df_analise['pessoa feliz'].astype(int)

model = tree.DecisionTreeClassifier(random_state=42, min_samples_leaf=5)
model.fit(X, y)

reg = linear_model.LogisticRegression(random_state=42,penalty=None, fit_intercept=True)
reg.fit(X, y)

nb = naive_bayes.GaussianNB()
nb.fit(X, y)

# %%

model_predict = model.predict(X)

df_predict = df_analise[['pessoa feliz']]
df_predict['predict_arvore'] = model_predict
df_predict['proba_arvore'] = model.predict_proba(X)[:,1]
df_predict['predict_reg'] = reg.predict(X)
df_predict['proba_reg'] = reg.predict_proba(X)[:,1]
df_predict['predict_nb'] = nb.predict(X)
df_predict['proba_nb'] = nb.predict_proba(X)[:,1]

# %%
from sklearn import metrics

acc_arvore = metrics.accuracy_score(list(df_predict['pessoa feliz']), list(df_predict['predict_arvore']))
precisao_arvore = metrics.precision_score(list(df_predict['pessoa feliz']), list(df_predict['predict_arvore']))
recall_arvore = metrics.recall_score(list(df_predict['pessoa feliz']), list(df_predict['predict_arvore']))
roc_arvore = metrics.roc_curve(list(df_predict['pessoa feliz']), list(df_predict['proba_arvore']))
roc_auc_arvore = metrics.roc_auc_score(list(df_predict['pessoa feliz']), list(df_predict['proba_arvore']))

acc_nb = metrics.accuracy_score(list(df_predict['pessoa feliz']), list(df_predict['predict_nb']))
precisao_nb = metrics.precision_score(list(df_predict['pessoa feliz']), list(df_predict['predict_nb']))
recall_nb = metrics.recall_score(list(df_predict['pessoa feliz']), list(df_predict['predict_nb']))
roc_nb = metrics.roc_curve(list(df_predict['pessoa feliz']), list(df_predict['proba_nb']))
roc_auc_nb = metrics.roc_auc_score(list(df_predict['pessoa feliz']), list(df_predict['proba_nb']))

acc_reg = metrics.accuracy_score(list(df_predict['pessoa feliz']), list(df_predict['predict_reg']))
precisao_reg = metrics.precision_score(list(df_predict['pessoa feliz']), list(df_predict['predict_reg']))
recall_reg = metrics.recall_score(list(df_predict['pessoa feliz']), list(df_predict['predict_reg']))
roc_reg = metrics.roc_curve(list(df_predict['pessoa feliz']), list(df_predict['proba_reg']))
roc_auc_reg = metrics.roc_auc_score(list(df_predict['pessoa feliz']), list(df_predict['proba_reg']))
# %%
import matplotlib.pyplot as plt
plt.figure(dpi=400)

plt.plot(roc_arvore[0], roc_arvore[1])
plt.grid(True)
plt.title('Curva ROC Arvore')
plt.xlabel('1 - Especificidade')
plt.ylabel('Sensibilidade (Recall)')
plt.text(0.6, 0.2, f'AUC = {roc_auc_arvore:.2f}', fontsize=12)
# %%
plt.plot(roc_nb[0], roc_nb[1])
plt.grid(True)
plt.title('Curva ROC Naive')
plt.xlabel('1 - Especificidade')
plt.ylabel('Sensibilidade (Recall)')
plt.text(0.6, 0.2, f'AUC = {roc_auc_nb:.2f}', fontsize=12)
# %%
plt.plot(roc_reg[0], roc_reg[1])
plt.grid(True)
plt.title('Curva ROC Regressão Logística')
plt.xlabel('1 - Especificidade')
plt.ylabel('Sensibilidade (Recall)')
plt.text(0.6, 0.2, f'AUC = {roc_auc_reg:.2f}', fontsize=12)

# %%
pd.Series({'model': reg, 'features': features}).to_pickle('model_feliz.pkl')
# %%
