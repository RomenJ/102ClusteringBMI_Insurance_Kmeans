"""
Este script carga datos de un archivo CSV que contiene información sobre edad, BMI (Índice de Masa Corporal) y cargos de seguro.
Luego, realiza análisis de clustering utilizando el algoritmo KMeans para agrupar los datos en clusters basados en la edad y el BMI.
Kaggle insurance dataset.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def main():
    """
    Función principal del script.
    """
    try:
        # Cargar y visualizar los datos
        data = pd.read_csv('insurance.csv', usecols=['age', 'bmi', 'charges'])
        print("Primeras 5 filas de los datos:")
        print(data.head())

        print("\nInformación de los datos:")
        print(data.info())

        # Eliminar filas con valores nulos
        data.dropna(inplace=True)

        # Visualización de dispersión de los datos
        sns.scatterplot(data=data, x="age", y="bmi", hue="charges")
        plt.title('Gráfico de dispersión de edad vs. BMI con cargos')
        plt.xlabel('Edad')
        plt.ylabel('BMI')
        plt.show()

        # Normalización de datos
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        # Determinación del número óptimo de clusters utilizando el método de Silhouette
        silhouette_scores = []
        for k in range(2, 11):
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(data_scaled)
            score = silhouette_score(data_scaled, kmeans.labels_)
            silhouette_scores.append(score)

        # Visualización del coeficiente de Silhouette para diferentes valores de k
        plt.plot(range(2, 11), silhouette_scores, marker='o')
        plt.xlabel('Número de clusters')
        plt.ylabel('Coeficiente de silhouette')
        plt.title('Selección del número óptimo de clusters')
        plt.show()

        # Entrenamiento del modelo KMeans con el número óptimo de clusters
        optimal_k = 3
        kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
        kmeans.fit(data_scaled)

        # Visualización de los clusters y centroides
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_

        plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=labels, s=50, cmap='spring')
        plt.scatter(centroids[:, 0], centroids[:, 1], marker="D", s=200, c='red')
        plt.xlabel('Edad (Estandarizada)')
        plt.ylabel('BMI (Estandarizado)')
        plt.title('Clusters y centroides obtenidos por KMeans')
        plt.show()

    except FileNotFoundError:
        print("El archivo CSV no fue encontrado.")
    except Exception as e:
        print("Se ha producido un error:", str(e))

if __name__ == "__main__":
    main()
"""""
Basado en
https://medium.com/latinxinai/tutorial-del-algoritmo-k-means-en-python-d8055751e2f3
"""""

