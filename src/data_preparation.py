import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def prepare_features(df: pd.DataFrame, target_column: str = "Churn"):
    df = df.copy()


    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])


    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    y = df[target_column]
    X = df.drop(columns=[target_column])


    if not pd.api.types.is_numeric_dtype(y):
        y = y.map({"No": 0, "Yes": 1}).fillna(y)

    X = pd.get_dummies(X, drop_first=True)
    X = X.fillna(0)

    return X, y.astype(int)