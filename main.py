import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('/content/FuelConsumptionCo2.csv')
x=df.iloc[:,4].values.reshape(-1,1)
y=df.iloc[:,-1].values.reshape(-1,1)
from sklearn.model_selection import train_test_split
train_x , test_x ,train_y , test_y=train_test_split(x,y,test_size=0.2 , random_state=0)
from sklearn.linear_model import LinearRegression
Lin=LinearRegression()
Lin.fit(train_x,train_y)
pred_y=Lin.predict(test_x)
np.concatenate((test_y,pred_y), axis=1)
np.absolute(test_y-pred_y)
np.absolute(test_y-pred_y).mean()
Lin.predict([[1.6]])

from sklearn import metrics
metrics.mean_absolute_error(test_y, pred_y)
metrics.mean_squared_error(test_y, pred_y)

#Graph
plt.scatter(test_x, test_y,  color='black')
plt.plot(test_x, pred_y, color='red', linewidth=3)
plt.title("CO2 emission vs Engine size")
plt.xlabel("Engine Size",color='blue')
plt.ylabel("Carbon Dioxide Emission",color='green')
plt.xticks(())
plt.yticks(())
