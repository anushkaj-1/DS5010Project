import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt


def remove_highly_correlated_columns(df, threshold=0.95):
    """
    Removes columns from DataFrame df that have a correlation higher than the specified threshold.
    """
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(to_drop, axis=1, inplace=False)


def apply_pca(df, n_components=None):
    """
    Applies PCA to the DataFrame df and returns transformed DataFrame.
    n_components specifies the number of components to keep.
    """
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(df_scaled)
    return pd.DataFrame(data=principal_components)


def handle_missing_values(df, strategy='mean'):
    """
    Handles missing values in the DataFrame df. The default strategy is to replace missing values with the mean.
    Other strategies can be 'median', 'mode', or 'drop', where 'drop' will remove rows with missing values.
    """
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'mean':
        return df.fillna(df.mean())
    elif strategy == 'median':
        return df.fillna(df.median())
    elif strategy == 'mode':
        return df.fillna(df.mode().iloc[0])
    else:
        raise ValueError("Unsupported strategy. Choose from 'mean', 'median', 'mode', or 'drop'.")


def remove_columns_with_high_missing_values(df, threshold=0.7):
    """
    Removes columns from DataFrame df that have more than a certain percentage (threshold) of missing values.
    """
    missing_ratio = df.isnull().mean()
    columns_to_drop = missing_ratio[missing_ratio > threshold].index
    return df.drop(columns=columns_to_drop)


def encode_categorical_variables(df, max_categories_for_onehot=10):
    """
    Encodes categorical variables using OneHotEncoder if they have less than or equal to 'max_categories_for_onehot'
    categories. Otherwise, uses LabelEncoder.
    Returns the modified DataFrame and the encoders used for each column (for potential inverse transformations).
    """
    encoders = {}
    for col in df.select_dtypes(include=['object', 'category']).columns:
        if df[col].nunique() <= max_categories_for_onehot:
            encoder = OneHotEncoder(sparse=False)
            transformed_data = encoder.fit_transform(df[[col]])
            df = pd.concat(
                [df.drop(columns=[col]), pd.DataFrame(transformed_data, columns=encoder.get_feature_names_out([col]))],
                axis=1)
        else:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col])
        encoders[col] = encoder
    return df, encoders


def remove_features_by_permutation_importance(df, target_column, model=None, importance_threshold=0.01):
    """
    Removes features from X_train and X_val based on permutation importance using the provided model.
    Features with an importance score lower than the importance_threshold are removed.

    Parameters:
    - X_train, y_train: Training data and labels.
    - X_val, y_val: Validation data and labels.
    - model: A fitted model that supports feature_importances_ or has a coef_ attribute.
             Defaults to RandomForestClassifier if None.
    - importance_threshold: The threshold for removing features based on importance score.

    Returns:
    - X_train_reduced, X_val_reduced: Dataframes with less important features removed.
    - important_features: List of important features retained.
    """

    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    if model is None:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

    result = permutation_importance(model, X_val, y_val, n_repeats=10, random_state=42, n_jobs=-1)
    important_features = [X_train.columns[i] for i in result.importances_mean.argsort()[::-1] if
                          result.importances_mean[i] - 2 * result.importances_std[i] > importance_threshold]

    X_train_reduced = X_train[important_features]
    X_val_reduced = X_val[important_features]

    return X_train_reduced, X_val_reduced, important_features


def optimal_number_of_components(df):
    pca = PCA().fit(df)
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

    plt.figure(figsize=(8, 5))
    plt.plot(cumulative_variance_ratio, marker='o', linestyle='--')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance by Number of Components')

    # Draw line at 95% of explained variance
    plt.axhline(y=0.95, color='r', linestyle='-')
    plt.text(0.5, 0.85, '95% cut-off threshold', color='red', fontsize=16)

    plt.show()

    n_components = np.where(cumulative_variance_ratio >= 0.95)[0][0] + 1
    return n_components

