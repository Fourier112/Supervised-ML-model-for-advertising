import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#I read the CSV file into a dataframe, and removed an unnecessary column. Next, I assigned my inputs and targets to a dataframe and series
#respectively.
data = pd.read_csv(r"C:\Users\quadr\OneDrive\Documents\csv_files\Advertising.csv")
data = data.iloc[:, 1:]
X = data[["TV"]]
y = data["Sales"]

#I first used the linear regression model from scikit-learn to fit the model.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)   #the predictions of the linear regression model.

def compute_cost(x, y, w, b):
    """
    x -> the training inputs, y -> the targets, w -> the current weight of the model and b -> the bias.
    This function uses the squared error cost function to compute the cost of a function's model at specific values for w and b. 
    """
    m = len(x)
    f_wb = (w * x) + b
    sq_errors = (f_wb - y) ** 2
    return (sum(sq_errors)) / (2 * m)

def compute_gradient(x, y, w, b):
    """
    Calculates the gradient of the cost function with respect to w and b.
    """
    m = len(x)
    f_wb = (w * x) + b
    error = f_wb - y
    dj_dw = (sum(error * x)) / m
    dj_db = (sum(error)) / m
    return dj_dw, dj_db

def batch_gradient_descent(X, y, w_init, b_init, lr, epochs):
    w, b = w_init, b_init
    history = []
    for i in range(epochs):
        dj_dw, dj_db = compute_gradient(X, y, w, b)
        w = w - (lr * dj_dw)
        b = b - (lr * dj_db)
        J = compute_cost(X, y, w, b)
        history.append((i, J, w, b))
        if i % 100 == 0:
            print(f'Epoch: {i} | w: {w:.4e} | b: {b:.4e} | cost: {J:.4f}')
    return w, b, history
 
X_train_np = X_train.to_numpy().reshape(-1)
y_train_np = y_train.to_numpy()
X_test_np = X_test.to_numpy().reshape(-1)
y_test_np = y_test.to_numpy()

w_final, b_final, history = batch_gradient_descent(X_train_np, y_train_np, w_init=0.0, b_init=0.0, lr=0.000001, epochs=1500)
#print(f"w_final: {w_final:.2f}, b_final: {b_final:.2f}")

y_gradient_descent = (w_final * X_test_np) + b_final
test_cost = compute_cost(X_test_np, y_test_np, w_final, b_final)
#print(f"Test cost: {test_cost:.2f}")

Epochs = [epoch for epoch, cost, weight, bias in history]
Costs = [cost for epoch, cost, weight, bias in history]
Weights = [weight for epoch, cost, weight, bias in history]
Biases = [bias for epoch, cost, weight, bias in history] 

plt.plot(Epochs, Costs)
plt.xlabel("Epochs")
plt.ylabel("Training costs")
plt.title("Learning Curve")
plt.show()
