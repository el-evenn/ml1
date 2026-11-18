#A simple ml model to predict the value of y given x using linear regression
from pathlib import Path
import numpy as np #numeric arrays
import matplotlib.pyplot as plt #plotting
from sklearn.model_selection import train_test_split    #splitting data into train and test sets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#define input and output data
x = np.array([1,2,3,4,5,6,7,8,9,10]).reshape(-1,1)
y = np.array([0.7,1.1,1.7,2.0,2.8,3.0,3.8,4.1,4.7,5.0], dtype=float)

#hold out two examples before fittig the model
x_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
    )

#learn the relationship between x and y using linear regression
model = LinearRegression()
model.fit(x_train, y_train)

#predict the value of y for the test set
predictions = model.predict(X_test)
print("Synthetic teachin data: 8 training examples, 2 test examples.")
for value, actual, predicted in zip (X_test.ravel(), y_test, predictions):
    print(f"Input: {value:g} | Actual: {actual:g}| Predicted: {predicted:.2f}")
print(f"Mean squared error: {mean_squared_error(y_test, predictions):.4f}(squared target units)")
print(f"Prediction for 6.5 daily light hours: {model.predict([[6.5]])[0]:.2f} weekly growth (cm)")

 #visualization
grid = np.linspace(x.min(), x.max(), 100).reshape(-1,1)
plt.scatter(x_train.ravel(), y_train, label="Training examples")
plt.scatter(X_test.ravel(), y_test, label="Test examples", marker="s")
plt.scatter(X_test.ravel(), predictions, label = "Test predictions", marker="X", s= 80)
plt.plot(grid.ravel(), model.predict(grid), label="Learned line", color="gray")
plt.xlabel("daily light hours")
plt.ylabel("weekly growth (cm)")
plt.title("Plant Growth"+"(synthetic data)")
plt.legend()