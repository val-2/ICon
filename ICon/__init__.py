import pathlib

import pandas as pd
from sklearnex import patch_sklearn

patch_sklearn()

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.tree import DecisionTreeRegressor

# Read the CSV file
original_df = pd.read_csv(pathlib.Path(__file__).parent.parent / 'used_cars.csv')


def supervised_learning(df, target_column="Price"):
    # Prendere le colonne categoriche e trasformarle in colonne numeriche facendo il one-hot encoding
    # In python gli alberi di decisione non supportano feature categoriche
    preprocessed_df = preprocessing(df)

    print(preprocessed_df)

    X = preprocessed_df.drop(target_column, axis=1)
    y = preprocessed_df[target_column]

    # Scelte tre metriche di scoring, ma prima e terza simili, todo non so se equivalgono
    scorings = ['neg_root_mean_squared_error', 'neg_mean_absolute_error', 'neg_mean_squared_error']

    results = {s: {"decision_tree": {"scores": None, "params": None},
               "random_forest": {"scores": None, "params": None},
               "gradient_boosted_trees": {"scores": None, "params": None}} for s in scorings }

    dtr_param_grid = {
        'criterion': ["squared_error", "friedman_mse", "absolute_error", "poisson"],
        'splitter': ['best'],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 5],
    }

    rfr_param_grid = {
        'n_estimators': [100, 200],
        'criterion': ["squared_error", "friedman_mse", "poisson"],  # "absolute_error" è troppo lento
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 5],
    }

    # Best params: {'criterion': 'poisson', 'max_depth': None, 'min_samples_leaf': 5, 'min_samples_split': 2, 'n_estimators': 200}

    gbr_param_grid = {
        'n_estimators': [100, 200],
        'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
        'learning_rate': [0.01, 0.1, 0.5],
        'criterion': ['friedman_mse', 'squared_error'],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 5],
    }

    # Perform 5-fold cross-validation, repeated 3 times, and print the average score
    cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=42)
    # TODO fonty per la grid search fa solo cv=5 che come se fosse n_repeats=1

    # Addestrare ogni modello per ogni metrica di scoring
    for scoring in scorings:
        baseline_score = calculate_baseline_score(df, target_column, scoring)
        print(f"Baseline for scoring metric {scoring}: ", baseline_score)

        dtc = DecisionTreeRegressor(random_state=42)
        print(f"Training decision tree with scoring metric {scoring}")
        results[scoring]["decision_tree"] = train_model(dtc, dtr_param_grid, X, y, cv, scoring)

        rfc = RandomForestRegressor(random_state=42)
        print(f"Training random forest with scoring metric {scoring}")
        results[scoring]["random_forest"] = train_model(rfc, rfr_param_grid, X, y, cv, scoring)

        gbt = GradientBoostingRegressor(random_state=42)
        print(f"Training gradient boosted model with scoring metric {scoring}")
        results[scoring]["gradient_boosted_trees"] = train_model(gbt, gbr_param_grid, X, y, cv, scoring)

    # TODO fare il plot dei risultati




def train_model(model, param_grid, X, y, cv, scoring):
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=-1, verbose=2)
    grid_search.fit(X, y)

    # scores = cross_val_score(ctf, X, y, cv=cv, scoring=scoring)
    # grid_search.fit implicitamente usa cross_val_score e quindi non abbiamo bisogno di ripeterlo
    # Gli score ottenuti usando cross_val_score con i migliori iperparametri sono già restituiti da grid_search
    scores = grid_search.best_score_
    params = grid_search.best_params_
    print("Scores:", scores)
    print("Best params:", params)

    model_results = {"scores": scores, "params": params}

    return model_results


def calculate_baseline_score(df, target_column, scoring):
    # Per vedere se la loss del modello è migliore di quella che si potrebbe ottenere con una predizione banale, come media o mediana
    if scoring == "neg_root_mean_squared_error":
        # Per loss quadratiche si usa la media
        baseline_prediction = df[target_column].mean()
        baseline_error = - (df[target_column] - baseline_prediction).pow(2).mean() ** 0.5
    elif scoring == "neg_mean_squared_error":
        baseline_prediction = df[target_column].mean()
        baseline_error = - (df[target_column] - baseline_prediction).pow(2).sum().mean()
    elif scoring == "neg_mean_absolute_error":
        # Per loss assolute si usa la mediana
        baseline_prediction = df[target_column].median()
        baseline_error = - (df[target_column] - baseline_prediction).abs().mean()
    else:
        raise ValueError("Invalid scoring metric")

    return baseline_error


def preprocessing(df):
    df = df.copy()
    # Per non avere valori nulli nella colonna Service History
    df['Service history'] = df['Service history'].fillna("Unavailable")
    df['Service history'] = df['Service history'].map({'Full': True, 'Unavailable': False})
    # Prendere le colonne categoriche e trasformarle in colonne numeriche facendo il one-hot encoding
    # In python gli alberi di decisione non supportano feature categoriche
    df = df.drop("title", axis=1)

    # Eliminare la prima colonna che è l'indice
    df = df.drop(df.columns[0], axis=1)

    # Prendi la cilindrata dell'auto e trasformala in un numero
    df['Engine'] = df['Engine'].str.replace('L', '').astype(float)

    df['Gearbox'] = df['Gearbox'].map({'Automatic': True, 'Manual': False})
    df = df.rename(columns={'Gearbox': 'Gearbox Automatic'})

    df['Emission Class'] = pd.to_numeric(df['Emission Class'].str.replace('Euro ', ''), errors='coerce')

    df = pd.get_dummies(df, columns=['Fuel type', 'Body type'])

    # Algoritmi necessitano di valori non nulli
    df = df.dropna()

    return df


if __name__ == "__main__":
    # Print the DataFrame
    print(original_df)

    supervised_learning(original_df)
