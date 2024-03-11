import pandas as pd
import pathlib
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


# Read the CSV file
df = pd.read_csv(pathlib.Path(__file__).parent.parent / 'used_cars.csv')

# Pre-processing

# Per non avere valori nulli nella colonna Service History
df['Service history'] = df['Service history'].fillna("Unavailable")

def supervised_learning(df):
    target_column = "Price"

    # Prendere le colonne categoriche e trasformarle in colonne numeriche facendo il one-hot encoding
    # In python gli alberi di decisione non supportano feature categoriche
    df = df.drop("title", axis=1)
    df_encoded = pd.get_dummies(df, columns=['Fuel type', 'Body type', 'Engine', 'Gearbox', 'Emission Class', 'Service history'])

    X = df_encoded.drop(target_column, axis=1)
    y = df_encoded[target_column]

    # Split the data into a training and a test set
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # Create a decision tree classifier
    ctf = DecisionTreeClassifier(random_state=1)

    # kf = KFold(n_splits=5, shuffle=True, random_state=42)  # Set random_state for reproducibility

    # Perform 5-fold cross-validation, repeated 3 times, and print the average score
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
    scoring = ['neg_root_mean_squared_error', 'neg_mean_absolute_error', 'neg_mean_squared_error']
    scores = cross_val_score(ctf, X, y, cv=cv, scoring=scoring[1])
    print("Average Cross-Validation Score:", scores.mean())


# Print the DataFrame
print(df)


supervised_learning(df)
