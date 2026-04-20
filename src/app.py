"""Proyecto 4Geeks: analisis de sentimientos con Naive Bayes.

Este script reproduce el flujo principal del notebook:
1. Carga y limpieza del dataset.
2. Vectorizacion del texto con CountVectorizer.
3. Comparacion de GaussianNB, MultinomialNB y BernoulliNB.
4. Optimizacion con MultinomialNB y Random Forest.
5. Prueba de modelos alternativos.
6. Guardado del mejor modelo.
"""

from pathlib import Path
import pickle
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_URL = (
    "https://raw.githubusercontent.com/4GeeksAcademy/"
    "naive-bayes-project-tutorial/main/playstore_reviews.csv"
)
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "playstore_reviews.csv"
PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed" / "playstore_reviews_clean.csv"
RESULTS_PATH = ROOT_DIR / "data" / "processed" / "model_results.csv"
MODEL_PATH = ROOT_DIR / "models" / "sentiment_best_model.pkl"
NB_MODEL_PATH = ROOT_DIR / "models" / "sentiment_multinomial_nb.pkl"
RF_MODEL_PATH = ROOT_DIR / "models" / "sentiment_random_forest.pkl"


def load_data() -> pd.DataFrame:
    """Load the local dataset, falling back to the public URL."""
    if RAW_DATA_PATH.exists():
        return pd.read_csv(RAW_DATA_PATH)

    return pd.read_csv(DATA_URL)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Keep useful columns and normalize review text."""
    clean_df = df.copy()
    clean_df = clean_df.drop(columns=["package_name"])
    clean_df = clean_df.dropna(subset=["review", "polarity"])
    clean_df["review"] = clean_df["review"].astype(str).str.strip().str.lower()
    clean_df = clean_df[clean_df["review"].str.len() > 0]
    clean_df = clean_df.drop_duplicates().reset_index(drop=True)
    clean_df["review_length"] = clean_df["review"].str.len()
    clean_df["word_count"] = clean_df["review"].str.split().str.len()
    return clean_df


def get_metrics(y_true, y_pred) -> dict[str, float]:
    """Return the main classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def evaluate_model(name: str, model, x_train, x_test, y_train, y_test) -> dict[str, Any]:
    """Train one model and return its test metrics."""
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    return {"model": name, **get_metrics(y_test, predictions)}


def main() -> None:
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = load_data()
    clean_df = clean_data(df)
    clean_df.to_csv(PROCESSED_DATA_PATH, index=False)

    x_train_text, x_test_text, y_train, y_test = train_test_split(
        clean_df["review"],
        clean_df["polarity"],
        test_size=0.2,
        random_state=42,
        stratify=clean_df["polarity"],
    )

    vectorizer = CountVectorizer(stop_words="english")
    x_train = vectorizer.fit_transform(x_train_text)
    x_test = vectorizer.transform(x_test_text)

    results = []
    nb_models = {
        "GaussianNB": GaussianNB(),
        "MultinomialNB": MultinomialNB(),
        "BernoulliNB": BernoulliNB(),
    }

    for name, model in nb_models.items():
        if name == "GaussianNB":
            results.append(evaluate_model(name, model, x_train.toarray(), x_test.toarray(), y_train, y_test))
        else:
            results.append(evaluate_model(name, model, x_train, x_test, y_train, y_test))

    optimized_nb = GridSearchCV(
        Pipeline(
            [
                ("vectorizer", CountVectorizer(stop_words="english")),
                ("model", MultinomialNB()),
            ]
        ),
        param_grid={
            "vectorizer__ngram_range": [(1, 1), (1, 2)],
            "model__alpha": [0.1, 0.5, 1.0, 2.0],
        },
        scoring="f1",
        cv=5,
        n_jobs=-1,
    )
    optimized_nb.fit(x_train_text, y_train)
    nb_predictions = optimized_nb.predict(x_test_text)
    results.append({"model": "Optimized MultinomialNB", **get_metrics(y_test, nb_predictions)})

    random_forest = GridSearchCV(
        Pipeline(
            [
                ("vectorizer", CountVectorizer(stop_words="english")),
                ("model", RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced")),
            ]
        ),
        param_grid={
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 30],
            "model__min_samples_leaf": [1, 2],
        },
        scoring="f1",
        cv=3,
        n_jobs=-1,
    )
    random_forest.fit(x_train_text, y_train)
    rf_predictions = random_forest.predict(x_test_text)
    results.append({"model": "Optimized RandomForest", **get_metrics(y_test, rf_predictions)})

    alternative_models = {
        "LogisticRegression TF-IDF": Pipeline(
            [
                ("vectorizer", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        solver="liblinear",
                        random_state=42,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "LinearSVC TF-IDF": Pipeline(
            [
                ("vectorizer", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
                ("model", LinearSVC(random_state=42, class_weight="balanced")),
            ]
        ),
    }

    fitted_alternatives = {}
    for name, model in alternative_models.items():
        model.fit(x_train_text, y_train)
        fitted_alternatives[name] = model
        predictions = model.predict(x_test_text)
        results.append({"model": name, **get_metrics(y_test, predictions)})

    results_df = pd.DataFrame(results).sort_values(by="f1", ascending=False)
    results_df.to_csv(RESULTS_PATH, index=False)

    candidates = {
        "Optimized MultinomialNB": optimized_nb.best_estimator_,
        "Optimized RandomForest": random_forest.best_estimator_,
        **fitted_alternatives,
    }
    best_model_name = results_df.iloc[0]["model"]
    best_model = candidates.get(best_model_name, optimized_nb.best_estimator_)

    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(best_model, model_file)

    with NB_MODEL_PATH.open("wb") as model_file:
        pickle.dump(optimized_nb.best_estimator_, model_file)

    with RF_MODEL_PATH.open("wb") as model_file:
        pickle.dump(random_forest.best_estimator_, model_file)

    print("Dataset limpio guardado en:", PROCESSED_DATA_PATH)
    print("Metricas guardadas en:", RESULTS_PATH)
    print("Mejor modelo:", best_model_name)
    print(results_df.round(4).to_string(index=False))
    print("Modelo final guardado en:", MODEL_PATH)
    print("Naive Bayes optimizado guardado en:", NB_MODEL_PATH)
    print("Random Forest optimizado guardado en:", RF_MODEL_PATH)


if __name__ == "__main__":
    main()
