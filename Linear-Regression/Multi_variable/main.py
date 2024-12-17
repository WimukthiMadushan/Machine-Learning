import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

data  = pd.read_csv('Book2.csv')

model = LinearRegression()
X = data[['videos', 'days', 'subscribers']]
Y = data['views']
model.fit(X,Y)

LinearRegression()

new_input = pd.DataFrame([[45, 180, 3100]], columns=['videos', 'days', 'subscribers'])
pred_y = model.predict(new_input)

print(pred_y)
plt.show()






