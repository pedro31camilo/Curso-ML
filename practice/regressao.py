#%%
import pandas as pd

df = pd.read_excel('data/dados_cerveja_nota.xlsx')
df
# %%
from sklearn import linear_model
from sklearn import tree

X = df[['cerveja']]
y = df['nota']

reg = linear_model.LinearRegression()
reg.fit(X, y)

arvore_full = tree.DecisionTreeRegressor(random_state=42)
arvore_full.fit(X, y)

arvore_d2 = tree.DecisionTreeRegressor(random_state=42, max_depth=2)
arvore_d2.fit(X, y)
# %%
a, b = reg.intercept_, reg.coef_[0]
print(a, b)
# %%
predict_reg = reg.predict(X.drop_duplicates())
predict_reg

predict_arvore_full = arvore_full.predict(X.drop_duplicates())

predict_arvore_d2 = arvore_d2.predict(X.drop_duplicates())
#%%
import matplotlib.pyplot as plt

plt.figure(dpi=400)

plt.plot(X['cerveja'], y, 'o')
plt.grid(True)
plt.title('Relação cerveja X notas')
plt.xlabel('Cerveja')
plt.ylabel('Notas')
plt.plot(X.drop_duplicates()['cerveja'], predict_reg)
plt.plot(X.drop_duplicates()['cerveja'], predict_arvore_full)
plt.plot(X.drop_duplicates()['cerveja'], predict_arvore_d2)
plt.legend(['Observado', f'y = {a:.3f} + {b:.3f}x', 'Árvore Full', 'Árvore D2'])
# %%
