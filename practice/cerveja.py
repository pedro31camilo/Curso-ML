#%%
import pandas as pd

df = pd.read_excel('data/dados_cerveja.xlsx')
df
# %%
features = ['temperatura', 'copo', 'espuma', 'cor']
target = 'classe'
# %%
from sklearn import tree

model  = tree.DecisionTreeClassifier(random_state=42)

X = df[features]
y=df[target]

X = X.replace({
    'mud': 1, 'pint':2,
    'sim': 1, 'não': 0,
    'clara': 0, 'escura': 1
})

model.fit(X=X, y=y)

# %%
import matplotlib.pyplot as plt

plt.figure(dpi=400)

tree.plot_tree(model, feature_names=features, class_names=model.classes_, filled=True)
