#%%
import pandas as pd

df= pd.read_parquet('data/dados_clones.parquet')
df.columns
# %%
X = df.drop(columns=['Status ', 'General Jedi encarregado'])
y = df['Status ']
X.columns
# %%
X = X.replace({
    'Yoda':1, 'Shaak Ti':2,
    'Obi-Wan Kenobi': 3, 'Aayla Secura': 4,
    'Mace Windu': 5, 'Tipo 1': 1,
    'Tipo 2': 2, 'Tipo 3': 3, 'Tipo 4': 4,
    'Tipo 5': 5
})
X

#%%
from sklearn import tree

model = tree.DecisionTreeClassifier(random_state=42)

model.fit(X, y)

#%%
import matplotlib.pyplot as plt

plt.figure(dpi=400)

tree.plot_tree(model, feature_names=X.columns,class_names=model.classes_, filled=True, max_depth=3)
# %%
