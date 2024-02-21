import pandas as pd
import pathlib

# Read the CSV file
df = pd.read_csv(pathlib.Path(__file__).parent.parent / 'used_cars.csv')

# Pre-processing
# df = df.drop('Service history', axis=1)


# Print the DataFrame
print(df)
