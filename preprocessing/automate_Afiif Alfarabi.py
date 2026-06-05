import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


def preprocess(
    input_file="heart_disease_uci.csv",
    output_dir="preprocessing"
):

    print("=" * 60)
    print("AUTOMATED PREPROCESSING")
    print("=" * 60)

    # =====================================================
    # Load Dataset
    # =====================================================

    df = pd.read_csv(input_file)

    print(f"Dataset berhasil dimuat")
    print(f"Shape : {df.shape}")

    # =====================================================
    # Drop Kolom Tidak Relevan
    # =====================================================

    df = df.drop(
        columns=["id", "dataset"],
        errors="ignore"
    )

    print("\nKolom setelah pembersihan:")
    print(df.columns.tolist())

    # =====================================================
    # Pisahkan Fitur dan Target
    # =====================================================

    X = df.drop(columns=["num"])
    y = df["num"]

    # =====================================================
    # Train Test Split
    # =====================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTrain Shape :", X_train.shape)
    print("Test Shape  :", X_test.shape)

    # =====================================================
    # Missing Value Handling
    # =====================================================

    for col in X_train.columns:

        if X_train[col].isnull().sum() > 0:

            if X_train[col].dtype in ["float64", "int64"]:

                median_value = X_train[col].median()

                X_train[col] = X_train[col].fillna(
                    median_value
                )

                X_test[col] = X_test[col].fillna(
                    median_value
                )

            else:

                mode_value = X_train[col].mode()[0]

                X_train[col] = X_train[col].fillna(
                    mode_value
                )

                X_test[col] = X_test[col].fillna(
                    mode_value
                )

    print("\nMissing value handling selesai")

    # =====================================================
    # Encoding
    # =====================================================

    categorical_columns = (
        X_train
        .select_dtypes(include=["object"])
        .columns
        .tolist()
    )

    for col in categorical_columns:

        encoder = LabelEncoder()

        X_train[col] = encoder.fit_transform(
            X_train[col].astype(str)
        )

        X_test[col] = encoder.transform(
            X_test[col].astype(str)
        )

    print("Encoding selesai")

    # =====================================================
    # Outlier Handling
    # =====================================================

    outlier_columns = [
        "trestbps",
        "chol",
        "thalch",
        "oldpeak"
    ]

    for col in outlier_columns:

        if col in X_train.columns:

            Q1 = X_train[col].quantile(0.25)
            Q3 = X_train[col].quantile(0.75)

            IQR = Q3 - Q1

            lower_bound = Q1 - (1.5 * IQR)
            upper_bound = Q3 + (1.5 * IQR)

            X_train[col] = np.clip(
                X_train[col],
                lower_bound,
                upper_bound
            )

            X_test[col] = np.clip(
                X_test[col],
                lower_bound,
                upper_bound
            )

    print("Outlier handling selesai")

    # =====================================================
    # Standardisasi
    # =====================================================

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=X_train.columns
    )

    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=X_test.columns
    )

    # =====================================================
    # Gabungkan Kembali
    # =====================================================

    train_processed = pd.concat(
        [
            X_train_scaled,
            y_train.reset_index(drop=True)
        ],
        axis=1
    )

    test_processed = pd.concat(
        [
            X_test_scaled,
            y_test.reset_index(drop=True)
        ],
        axis=1
    )

    # =====================================================
    # Simpan Dataset
    # =====================================================

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    train_processed.to_csv(
        f"{output_dir}/train_preprocessed.csv",
        index=False
    )

    test_processed.to_csv(
        f"{output_dir}/test_preprocessed.csv",
        index=False
    )

    print("\nDataset berhasil disimpan")
    print(f"{output_dir}/train_preprocessed.csv")
    print(f"{output_dir}/test_preprocessed.csv")

    return train_processed, test_processed


if __name__ == "__main__":
    preprocess()