#Mean Absolute Error
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

df = pd.DataFrame({
    'hours': [8, 11, 9, 5, 7.5, 9.5, 10, 7, 9, 9.5, 8, 10.5, 9, 6.5, 9],
    'score': [56, 70, 51, 24, 30, 66, 48, 36, 42, 61, 39, 87, 73, 48, 46]
})
head = df.head()
#print(head)

y = np.array(df.score.values)
y_pred = 8 * df.hours.values - 15

#print(y)
#print(y_pred)

mae = mean_absolute_error(y, y_pred)
print(f"Mean absolute Error {mae}")

##Root Mean Squared Error
rmse = root_mean_squared_error(y, y_pred)
print(f"Root Mean Square Error {rmse}")

## R-Squared
rs = r2_score(y, y_pred)
print(f"R2 Score {rs}")
