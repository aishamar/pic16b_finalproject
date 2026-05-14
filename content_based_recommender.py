"""
Content-based music recommendation utilities for dataset.csv.

This module is tailored to the Spotify-style dataset with columns such as:
track_name, artists, album_name, track_genre, and audio features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


AUDIO_FEATURE_COLUMNS = [
    "popularity",
    "duration_ms",
    "explicit",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature",
]

KNN_TEXT_COLUMNS = ["artists", "album_name", "track_name", "track_genre"]
SVM_TEXT_COLUMNS = ["artists", "album_name", "track_name"]


def load_music_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load and lightly clean the provided music dataset.
    """
    df = pd.read_csv(csv_path)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    df = df.dropna(subset=["artists", "album_name", "track_name"]).copy()
    df = df.drop_duplicates(subset=["track_id"]).reset_index(drop=True)
    return df


def build_joined_text(
    df: pd.DataFrame,
    text_columns: list[str],
    output_column: str,
) -> pd.DataFrame:
    """
    Combine selected string columns into one normalized text field.
    """
    missing = [column for column in text_columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing text columns: {missing}")

    prepared = df.copy()
    prepared[output_column] = (
        prepared[text_columns]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return prepared


@dataclass
class MusicContentRecommender:
    """
    KNN recommender using text metadata plus standardized audio features.
    """

    item_frame: pd.DataFrame
    vectorizer: TfidfVectorizer
    scaler: StandardScaler
    model: NearestNeighbors
    content_matrix: csr_matrix

    @classmethod
    def fit(
        cls,
        df: pd.DataFrame,
        max_features: int = 12000,
        ngram_range: tuple[int, int] = (1, 2),
    ) -> "MusicContentRecommender":
        required = set(KNN_TEXT_COLUMNS + AUDIO_FEATURE_COLUMNS)
        missing = sorted(required.difference(df.columns))
        if missing:
            raise KeyError(f"Missing required columns: {missing}")

        working = build_joined_text(df, KNN_TEXT_COLUMNS, "content_text_knn")

        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=max_features,
            ngram_range=ngram_range,
        )
        text_matrix = vectorizer.fit_transform(working["content_text_knn"])

        scaler = StandardScaler()
        audio_matrix = scaler.fit_transform(
            working[AUDIO_FEATURE_COLUMNS].astype(float)
        )
        content_matrix = hstack([text_matrix, csr_matrix(audio_matrix)]).tocsr()

        model = NearestNeighbors(metric="cosine", algorithm="brute")
        model.fit(content_matrix)

        return cls(
            item_frame=working.reset_index(drop=True),
            vectorizer=vectorizer,
            scaler=scaler,
            model=model,
            content_matrix=content_matrix,
        )

    def recommend_songs(
        self,
        track_name: str,
        artist_name: str | None = None,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """
        Recommend tracks similar to a known song.
        """
        candidates = self.item_frame[
            self.item_frame["track_name"].astype(str).str.lower()
            == track_name.lower()
        ]

        if artist_name is not None:
            candidates = candidates[
                candidates["artists"]
                .astype(str)
                .str.lower()
                .str.contains(artist_name.lower(), na=False)
            ]

        if candidates.empty:
            raise ValueError(
                "Track not found. Check the exact track name and optional artist name."
            )

        source_index = int(candidates.index[0])
        distances, indices = self.model.kneighbors(
            self.content_matrix[source_index],
            n_neighbors=min(top_n + 1, len(self.item_frame)),
        )

        rows = []
        for idx, distance in zip(indices.flatten(), distances.flatten()):
            if idx == source_index:
                continue
            rows.append(
                {
                    "track_name": self.item_frame.iloc[idx]["track_name"],
                    "artists": self.item_frame.iloc[idx]["artists"],
                    "track_genre": self.item_frame.iloc[idx]["track_genre"],
                    "popularity": int(self.item_frame.iloc[idx]["popularity"]),
                    "similarity_score": round(1 - float(distance), 4),
                }
            )
            if len(rows) == top_n:
                break

        return pd.DataFrame(rows)

    def recommend_from_music_profile(
        self,
        query_text: str,
        audio_profile: Mapping[str, float] | None = None,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """
        Recommend tracks from a text preference plus an optional audio profile.
        """
        query_text_matrix = self.vectorizer.transform([query_text])

        if audio_profile is None:
            audio_frame = (
                self.item_frame[AUDIO_FEATURE_COLUMNS].median().to_frame().T
            )
        else:
            audio_frame = pd.DataFrame([audio_profile])[AUDIO_FEATURE_COLUMNS]

        query_audio_matrix = self.scaler.transform(audio_frame.astype(float))
        query_matrix = hstack(
            [query_text_matrix, csr_matrix(query_audio_matrix)]
        ).tocsr()

        distances, indices = self.model.kneighbors(
            query_matrix,
            n_neighbors=min(top_n, len(self.item_frame)),
        )

        rows = []
        for idx, distance in zip(indices.flatten(), distances.flatten()):
            rows.append(
                {
                    "track_name": self.item_frame.iloc[idx]["track_name"],
                    "artists": self.item_frame.iloc[idx]["artists"],
                    "track_genre": self.item_frame.iloc[idx]["track_genre"],
                    "similarity_score": round(1 - float(distance), 4),
                }
            )

        return pd.DataFrame(rows)


def train_svm_genre_classifier(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    max_features: int = 12000,
) -> dict[str, object]:
    """
    Train an SVM classifier to predict track_genre.
    """
    required = set(SVM_TEXT_COLUMNS + AUDIO_FEATURE_COLUMNS + ["track_genre"])
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    working = build_joined_text(df, SVM_TEXT_COLUMNS, "content_text_svm")
    genre_counts = working["track_genre"].value_counts()
    eligible_genres = genre_counts[genre_counts >= 2].index
    working = working[working["track_genre"].isin(eligible_genres)].reset_index(
        drop=True
    )

    if working["track_genre"].nunique() < 2:
        raise ValueError(
            "SVM genre training needs at least two genres with two or more rows each."
        )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        ngram_range=(1, 2),
    )
    text_matrix = vectorizer.fit_transform(working["content_text_svm"])

    scaler = StandardScaler()
    audio_matrix = scaler.fit_transform(
        working[AUDIO_FEATURE_COLUMNS].astype(float)
    )
    feature_matrix = hstack([text_matrix, csr_matrix(audio_matrix)]).tocsr()

    target = working["track_genre"]
    x_train, x_test, y_train, y_test = train_test_split(
        feature_matrix,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    model = LinearSVC(class_weight="balanced", random_state=random_state)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    return {
        "model": model,
        "vectorizer": vectorizer,
        "scaler": scaler,
        "x_test": x_test,
        "y_test": y_test,
        "predictions": predictions,
        "accuracy": accuracy_score(y_test, predictions),
        "classification_report": classification_report(
            y_test,
            predictions,
            zero_division=0,
        ),
    }
