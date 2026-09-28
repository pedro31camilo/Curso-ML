#%%

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel('data/dados_cerveja_nota.xlsx')
df.head()

df['Aprovado'] = (df['nota'] > 5).astype(int)
df.head(10)
# %%
plt.plot(df['cerveja'], df['Aprovado'], 'o', color='royalblue')
plt.grid(True)
plt.title('Cerveja Vs Aprovado')
plt.xlabel('Cerveja')
plt.ylabel('Aprovado')

# %%
from sklearn import linear_model
from sklearn import tree
from sklearn import naive_bayes

reg = linear_model.LogisticRegression(penalty=None, fit_intercept=True)
reg.fit(df[['cerveja']], df['Aprovado'])
reg_predict = reg.predict(df[['cerveja']].drop_duplicates())
reg_predict_proba = reg.predict_proba(df[['cerveja']].drop_duplicates())

arvore_full = tree.DecisionTreeClassifier(random_state=42)
arvore_full.fit(df[['cerveja']], df['Aprovado'])
arvore_full_predict = arvore_full.predict(df[['cerveja']].drop_duplicates())
arvore_full_predict_proba = arvore_full.predict_proba(df[['cerveja']].drop_duplicates())

arvore_d2 = tree.DecisionTreeClassifier(random_state=42, max_depth=2)
arvore_d2.fit(df[['cerveja']], df['Aprovado'])
arvore_d2_predict = arvore_d2.predict(df[['cerveja']].drop_duplicates())
arvore_d2_predict_proba = arvore_d2.predict_proba(df[['cerveja']].drop_duplicates())


nb = naive_bayes.GaussianNB()
nb.fit(df[['cerveja']], df['Aprovado'])
nb_predict = nb.predict(df[['cerveja']].drop_duplicates())
nb_predict_proba = nb.predict_proba(df[['cerveja']].drop_duplicates())

plt.plot(df[['cerveja']].drop_duplicates(), reg_predict, 'o', color='royalblue')
plt.plot(df[['cerveja']].drop_duplicates(), reg_predict, color='tomato')
plt.plot(df[['cerveja']].drop_duplicates(), reg_predict_proba[:, 1], color='green')

#plt.plot(df[['cerveja']].drop_duplicates(), arvore_full_predict, color='yellow')
#plt.plot(df[['cerveja']].drop_duplicates(), arvore_full_predict_proba[:, 1], color='magenta')
plt.plot(df[['cerveja']].drop_duplicates(), arvore_d2_predict, color='orange')
plt.plot(df[['cerveja']].drop_duplicates(), arvore_d2_predict_proba[:, 1], color='purple')
plt.plot(df[['cerveja']].drop_duplicates(), nb_predict, color='brown')
plt.plot(df[['cerveja']].drop_duplicates(), nb_predict_proba[:, 1], color='black')

plt.grid(True)
plt.title('Cerveja Vs Aprovado')
plt.xlabel('Cerveja')
plt.ylabel('Aprovado')
plt.hlines(0.5, xmin=1, xmax=9, linestyle='--', color='black')
plt.legend(['Observação', 
            'Reg Predict', 
            'Reg Proba', 
            #'Arvore Predict', 
            #'Arvore Proba', 
            'Arvore D2 Predict', 
            'Arvore D2 Proba', 
            'NB Predict', 
            'NB Proba'])
# %%
