import numpy as np
from sklearn.cluster import KMeans

# Dados: Altura e Peso
dados = np.array([
    [160, 55],  # Pessoa A
    [170, 65],  # Pessoa B
    [175, 70],  # Pessoa C
    [180, 80],  # Pessoa D
    [150, 50]   # Pessoa E
])

# Definir o número de clusters
n_clusters = 2

# Aplicar o modelo KMeans
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(dados)

# Obter os rótulos e centróides
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

print("Rótulos dos clusters:", labels)
print("Centróides:", centroids)

# Interpretação dos resultados
print("\nInterpretação dos clusters:")
print(f"Cluster 0: Altura média = {centroids[0][0]:.1f}, Peso médio = {centroids[0][1]:.1f}")
print(f"Cluster 1: Altura média = {centroids[1][0]:.1f}, Peso médio = {centroids[1][1]:.1f}")
