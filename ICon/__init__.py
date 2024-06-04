import pathlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.tree import DecisionTreeRegressor

# Read the CSV file
original_df = pd.read_csv(pathlib.Path(__file__).parent.parent / 'used_cars.csv')




def plot_results(results, scorings):
    for scoring in scorings:
        fig, ax = plt.subplots(figsize=(10, 6))

        models = ['decision_tree', 'random_forest', 'gradient_boosted_trees']
        scores = [results[scoring][model]["scores"] for model in models]

        # Extract mean scores for plotting
        mean_scores = [np.mean(score) for score in scores]

        ax.bar(models, mean_scores, color=['blue', 'green', 'red'])

        ax.set_title(f'Performance Comparison ({scoring})')
        ax.set_xlabel('Models')
        ax.set_ylabel('Mean Score')
        ax.set_ylim([min(mean_scores)*0.9, max(mean_scores)*1.1])  # Adjust y-axis for better visualization
        ax.grid(True)

        # Adding text for each bar
        for i, v in enumerate(mean_scores):
            ax.text(i, v + (max(mean_scores)*0.01), f"{v:.2f}", ha='center', va='bottom')

        plt.savefig(f"performance_comparison_{scoring}.png")


def plot_learning_curve(estimator, title, X, y, scoring, cv, n_jobs=None, train_sizes=np.linspace(0.1, 1.0, 10)):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel(scoring)

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring=scoring
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    plt.savefig(f"{title}.png")
    return plt


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

    # Prendere solo la marca dell'auto
    df['title'] = df['title'].apply(lambda x: x.split()[0] if isinstance(x, str) else x)

    # Eliminare la prima colonna che è l'indice
    df = df.drop(df.columns[0], axis=1)

    # Prendi la cilindrata dell'auto e trasformala in un numero
    df['Engine'] = df['Engine'].str.replace('L', '').astype(float)

    df['Gearbox'] = df['Gearbox'].map({'Automatic': True, 'Manual': False})
    df = df.rename(columns={'Gearbox': 'Gearbox Automatic'})

    df['Emission Class'] = pd.to_numeric(df['Emission Class'].str.replace('Euro ', ''), errors='coerce')

    df = pd.get_dummies(df, columns=['Fuel type', 'Body type', 'title'])

    # Algoritmi necessitano di valori non nulli
    df = df.dropna()

    return df


if __name__ == "__main__":
    # Print the DataFrame
    print(original_df)

    supervised_learning(original_df)
