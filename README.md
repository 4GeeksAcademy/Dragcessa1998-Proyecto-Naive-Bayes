# Proyecto Naive Bayes: Análisis de Sentimientos en Reseñas de Google Play

Este proyecto desarrolla un modelo de Machine Learning para clasificar reseñas de aplicaciones de Google Play como positivas o negativas. El objetivo principal es aplicar técnicas de procesamiento de texto y comparar diferentes modelos de clasificación, prestando especial atención a las implementaciones de Naive Bayes.

El proyecto forma parte de una práctica de clasificación supervisada, donde se trabaja con datos textuales, limpieza de información, vectorización de palabras, entrenamiento de modelos, evaluación de métricas y almacenamiento del modelo final.

## Objetivo del Proyecto

El objetivo es construir un clasificador capaz de predecir la polaridad de una reseña de Google Play:

- `0`: reseña negativa
- `1`: reseña positiva

Para ello se utiliza la variable `review`, que contiene el texto escrito por los usuarios. La variable `package_name` se elimina porque el sentimiento debe depender del contenido del comentario y no de la aplicación a la que pertenece.

## Dataset

El conjunto de datos utilizado contiene reseñas de aplicaciones móviles publicadas en Google Play.

Variables originales:

| Variable | Descripción |
| --- | --- |
| `package_name` | Nombre del paquete o aplicación móvil |
| `review` | Comentario realizado por el usuario |
| `polarity` | Variable objetivo: 0 para negativo y 1 para positivo |

Después de la limpieza, el proyecto trabaja principalmente con:

| Variable | Descripción |
| --- | --- |
| `review` | Texto normalizado de la reseña |
| `polarity` | Sentimiento asociado a la reseña |
| `review_length` | Longitud del comentario en caracteres |
| `word_count` | Número de palabras de la reseña |

## Estructura del Proyecto

```text
Dragcessa1998-Proyecto-Naive-Bayes-main/
│
├── data/
│   ├── raw/
│   │   └── playstore_reviews.csv
│   ├── processed/
│   │   ├── playstore_reviews_clean.csv
│   │   └── model_results.csv
│   └── interim/
│
├── models/
│   ├── sentiment_best_model.pkl
│   ├── sentiment_multinomial_nb.pkl
│   └── sentiment_random_forest.pkl
│
├── src/
│   ├── app.py
│   ├── explore.ipynb
│   └── utils.py
│
├── requirements.txt
├── README.md
└── README.es.md
```

## Metodología

El desarrollo del proyecto se realizó siguiendo un flujo ordenado de ciencia de datos:

1. Carga del conjunto de datos.
2. Revisión inicial de variables, tipos de datos, valores nulos y distribución de la clase objetivo.
3. Eliminación de la variable `package_name`.
4. Limpieza del texto:
   - Conversión a minúsculas.
   - Eliminación de espacios iniciales y finales.
   - Eliminación de duplicados.
5. Análisis visual de la distribución de reseñas positivas y negativas.
6. Análisis de longitud de reseñas y número de palabras.
7. Vectorización del texto mediante `CountVectorizer`.
8. División del dataset en entrenamiento y prueba.
9. Entrenamiento de tres modelos Naive Bayes:
   - `GaussianNB`
   - `MultinomialNB`
   - `BernoulliNB`
10. Optimización del mejor Naive Bayes.
11. Comparación con Random Forest.
12. Evaluación de modelos alternativos como Logistic Regression y Linear SVC.
13. Guardado de los modelos entrenados.

## Modelos Evaluados

Se entrenaron y compararon distintos modelos de clasificación para determinar cuál ofrecía el mejor rendimiento sobre el conjunto de prueba.

| Modelo | Accuracy | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression + TF-IDF | 0.8539 | 0.8070 | 0.7541 | 0.7797 |
| Linear SVC + TF-IDF | 0.8427 | 0.8000 | 0.7213 | 0.7586 |
| MultinomialNB optimizado | 0.8371 | 0.8200 | 0.6721 | 0.7387 |
| Random Forest optimizado | 0.8202 | 0.7377 | 0.7377 | 0.7377 |
| MultinomialNB | 0.8146 | 0.8043 | 0.6066 | 0.6916 |
| GaussianNB | 0.7809 | 0.6897 | 0.6557 | 0.6723 |
| BernoulliNB | 0.7921 | 0.8750 | 0.4590 | 0.6022 |

## Resultados

El modelo `MultinomialNB` fue el mejor entre las tres implementaciones de Naive Bayes. Este resultado es coherente con el tipo de problema, ya que el texto fue transformado en una matriz de recuento de palabras mediante `CountVectorizer`.

`GaussianNB` no fue la mejor alternativa porque está pensado para variables continuas con una distribución aproximadamente normal. En este proyecto, las variables predictoras son recuentos de palabras, por lo que no se ajustan bien a ese supuesto.

`BernoulliNB` obtuvo un rendimiento inferior porque trabaja mejor con variables binarias, es decir, con presencia o ausencia de palabras, mientras que en este caso se conserva información sobre la frecuencia de los términos.

También se probó un modelo `RandomForestClassifier`, pero no logró superar al mejor modelo lineal. Finalmente, el mejor rendimiento global se obtuvo con `LogisticRegression` combinada con `TfidfVectorizer`.

## Conclusión

El proyecto demuestra que Naive Bayes es una técnica útil y eficiente para problemas de clasificación de texto. En particular, `MultinomialNB` resulta adecuado cuando las reseñas se representan como recuentos de palabras.

Sin embargo, al comparar con otros modelos estudiados, `LogisticRegression` con representación TF-IDF consiguió el mejor rendimiento general. Esto se debe a que TF-IDF pondera mejor la importancia de las palabras dentro del corpus, reduciendo el peso de términos demasiado frecuentes y resaltando aquellos que ayudan más a diferenciar entre reseñas positivas y negativas.

Por este motivo, se guardaron dos modelos principales:

- `sentiment_multinomial_nb.pkl`: mejor modelo Naive Bayes.
- `sentiment_best_model.pkl`: mejor modelo global del proyecto.

## Archivos Generados

| Archivo | Descripción |
| --- | --- |
| `data/raw/playstore_reviews.csv` | Dataset original |
| `data/processed/playstore_reviews_clean.csv` | Dataset limpio y procesado |
| `data/processed/model_results.csv` | Resultados comparativos de los modelos |
| `models/sentiment_multinomial_nb.pkl` | Modelo Naive Bayes optimizado |
| `models/sentiment_random_forest.pkl` | Modelo Random Forest optimizado |
| `models/sentiment_best_model.pkl` | Mejor modelo global |

## Cómo Ejecutar el Proyecto

Desde la raíz del proyecto, instala las dependencias necesarias:

```bash
pip install -r requirements.txt
```

Luego ejecuta el script principal:

```bash
python src/app.py
```

También puedes abrir y ejecutar el notebook:

```text
src/explore.ipynb
```

El notebook contiene el desarrollo completo del análisis, las gráficas, la comparación de modelos y las conclusiones del proyecto.

## Tecnologías Utilizadas

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook
- Pickle

## Estado del Proyecto

Proyecto finalizado correctamente.  
Incluye análisis exploratorio, limpieza de texto, entrenamiento de modelos, optimización, evaluación comparativa y almacenamiento del modelo final.


## Contributors

This template was built as part of the [Data Science and Machine Learning Bootcamp](https://4geeksacademy.com/us/coding-bootcamps/datascience-machine-learning) by 4Geeks Academy by [Alejandro Sanchez](https://twitter.com/alesanchezr) and many other contributors. Learn more about [4Geeks Academy BootCamp programs](https://4geeksacademy.com/us/programs) here.

Other templates and resources like this can be found on the school's GitHub page.
