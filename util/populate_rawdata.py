import pandas as pd
from sklearn.datasets import load_iris

df = pd.DataFrame(load_iris().data, columns=load_iris().feature_names)
df.to_csv("data/raw/iris.csv", index=False)
