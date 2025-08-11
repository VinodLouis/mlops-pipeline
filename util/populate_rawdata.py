import pandas as pd
from sklearn.datasets import load_iris

# Load the iris dataset
iris = load_iris()

# Create a DataFrame with feature data
df = pd.DataFrame(iris.data, columns=[
    "SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"
])

# Add Id column starting from 1
df.insert(0, "Id", range(1, len(df) + 1))

# Map numeric target to species names
species_map = dict(zip(range(3), iris.target_names))
df["Species"] = [f"Iris-{species_map[label].capitalize()}" for label in iris.target]

# Save to CSV
df.to_csv("data/raw/iris.csv", index=False)
