import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.ma.extras import polyfit
from sklearn.linear_model import LinearRegression


data = pd.read_csv('Book.csv')
plt.scatter(data.videos, data.views)
plt.xlabel('Videos')
plt.ylabel('Views')
#plot the diagram
#plt.show()

##Train Data
## Should make numpy Arrays

x = np.array(data.videos.values)
y = np.array(data.views.values)

##Create an object
model = LinearRegression()
##Train data
model.fit(x.reshape((-1,1)), y)

LinearRegression()

new_x = np.array([45]).reshape((-1, 1))
pred = model.predict(new_x)
#print(pred)

plt.scatter(data.videos, data.views, color='red')
m,c = polyfit(x,y,1)
plt.plot(x,m*x+c,color='blue')
plt.show()
print(m,c)



