import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  

#Task 1
#Prediction Using Supervised ML

dataset = pd.read_csv(r'D:\Android\Practice\student_scores.csv')
#print(dataset)
#print(dataset.head())
#print(dataset.shape)

dataset.plot(x='Hours', y='Scores', style='o')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()

X = dataset.iloc[:, :-1].values    
y = dataset.iloc[:, 1].values  

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression    
regressor = LinearRegression()    
regressor.fit(X_train, y_train)   
  
print("Training.......")

line = regressor.coef_*X+regressor.intercept_  
plt.scatter(X, y)  
plt.plot(X, line)
plt.show() 

print(X_test)   
y_pred = regressor.predict(X_test)  

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})    
print(df)

hours = [[10]]  
own_pred = regressor.predict(hours)  
print("Number of hours = {}".format(hours))  
print("Prediction Score = {}".format(own_pred[0]))  
