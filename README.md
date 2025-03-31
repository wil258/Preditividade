import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Exemplo de dados fictícios
# Cada linha representa uma luta com estatísticas dos lutadores
# 1 significa vitória do Lutador 1, 0 significa vitória do Lutador 2
dados = {
    'golpes_lutador1': [50, 60, 45, 70, 55],
    'golpes_lutador2': [40, 50, 55, 65, 60],
    'quedas_lutador1': [3, 2, 1, 4, 2],
    'quedas_lutador2': [1, 3, 2, 3, 4],
    'vitorias_lutador1': [10, 15, 20, 12, 8],
    'vitorias_lutador2': [12, 10, 18, 14, 9],
    'resultado': [1, 1, 0, 1, 0]  # 1 = Lutador 1 vence, 0 = Lutador 2 vence
}

df = pd.DataFrame(dados)

# Separação entre variáveis preditoras e variável alvo
X = df.drop(columns=['resultado'])
y = df['resultado']

# Divisão em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo de Machine Learning (Random Forest)
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Predição e avaliação
y_pred = modelo.predict(X_test)
acuracia = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo: {acuracia:.2f}')

# Exemplo de previsão com novos dados
novo_dado = np.array([[55, 48, 2, 3, 14, 10]])  # Estatísticas de uma nova luta
previsao = modelo.predict(novo_dado)
print(f'Previsão do vencedor: Lutador {previsao[0] + 1}')
