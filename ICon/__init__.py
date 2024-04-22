import pandas as pd
import pathlib
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


# Read the CSV file
df = pd.read_csv(pathlib.Path(__file__).parent.parent / 'used_cars.csv')

def supervised_learning(df, target_column="Price"):

    # Prendere le colonne categoriche e trasformarle in colonne numeriche facendo il one-hot encoding
    # In python gli alberi di decisione non supportano feature categoriche
    df_encoded = preprocessing(df)

    X = df_encoded.drop(target_column, axis=1)
    y = df_encoded[target_column]

    # # Calculate the baseline todo later
    # baseline = calculate_baseline(df, target_column, scorings[1])
    # print("Baseline:", baseline)

    # Split the data into a training and a test set
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    metrics2scores = {}

    # Create a decision tree classifier
    ctf = DecisionTreeClassifier(random_state=1)

    # Perform 5-fold cross-validation, repeated 3 times, and print the average score
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
    # Scelte tre metriche di scoring, ma prima e terza simili, todo non so se equivalgono
    scorings = ['neg_root_mean_squared_error', 'neg_mean_absolute_error', 'neg_mean_squared_error']

    # Addestrare un modello per ogni metrica di scoring
    for scoring in scorings:
        scores = cross_val_score(ctf, X, y, cv=cv, scoring=scoring)
        print("Average Cross-Validation Score:", scores.mean())
        metrics2scores[scoring] = scores




def calculate_baseline(df, target_column, scoring):
    # Per vedere se la loss del modello è migliore di quella che si potrebbe ottenere con una predizione banale, come media o mediana
    if scoring == "neg_root_mean_squared_error":
        # Per loss quadratiche si usa la media
        baseline_prediction = df[target_column].mean()
        baseline_prediction = - (df[target_column] - baseline_prediction).pow(2).mean().pow(0.5)
    elif scoring == "neg_mean_squared_error":
        baseline_prediction = df[target_column].mean()
        baseline_prediction = - (df[target_column] - baseline_prediction).pow(2).sum().mean()
    elif scoring == "neg_mean_absolute_error":
        # Per loss assolute si usa la mediana
        baseline_prediction = df[target_column].median()
        baseline_prediction = - (df[target_column] - baseline_prediction).abs().mean()
    else:
        raise ValueError("Invalid scoring metric")

    return baseline_prediction


def preprocessing(df):
    # Per non avere valori nulli nella colonna Service History
    df['Service history'] = df['Service history'].fillna("Unavailable")
    # Prendere le colonne categoriche e trasformarle in colonne numeriche facendo il one-hot encoding
    # In python gli alberi di decisione non supportano feature categoriche
    df = df.drop("title", axis=1)
    df_encoded = pd.get_dummies(df, columns=['Fuel type', 'Body type', 'Engine', 'Gearbox', 'Emission Class', 'Service history'])
    return df_encoded


if __name__ == "__main__":
    # Print the DataFrame
    print(df)


    supervised_learning(df)
