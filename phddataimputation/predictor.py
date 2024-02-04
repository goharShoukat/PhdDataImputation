from utils import PostProcessing, featureGeneration
import pandas as pd
from config import config1


path = "models/1/Model1-128NeuronsScaled"
x, y = featureGeneration(
    pd.read_csv("data/trainingData/M2_1hour_Gaps_10%_Missing.csv").dropna().to_numpy(),
    1,
    False,
)
foo = PostProcessing(path)
model = foo.load()
foo.normalise(x, y)
scaled_predictions = foo.predict(foo.X)
results = foo.denormalise(scaled_predictions)
