import numpy as np

class perceptron:
  def __init__(self, eta, epochs):
    self.weights = np.random.randn(3) * 1e-4 #small initialize small weight, random weights are multiplied with small number
    print(f"Intial weights before training: \n{self.weights}")
    self.eta = eta #learning rate
    self.epochs = epochs
  

  def activationFunction(self,inputs, weights):
    z = np.dot(inputs, weights) # z = w * x
    return np.where(z> 0 , 1, 0)

  def fit(self, X, y):
    self.X = X
    self.y = y

    X_with_bias = np.c_[self.X, -np.ones((len(X), 1))] # CONCATINATION
    print(f"X with bias: \n{X_with_bias}")


    for epoch in range(self.epochs):
      print("--"* 10)
      print(f"for epoch: \n{epoch}")
      print("--"* 10)

      y_hat = self.activationFunction(X_with_bias, self.weights)# forward propogation
      print(f"predicted value after forward pass: \n {y_hat}")
      self.error = self.y - y_hat
      print(f"error : \n{self.error}")

      self.weights = self.weights + self.eta * np.dot(self.error, X_with_bias) # weight updation, backward propogation
      print(f"updated weights after epoch: \n a{epoch}/{self.epochs}: \n{self.weights}")
      print("#####"*10)

  def predict(self, x):
    X_with_bias = np.c_[x, -np.ones((len(x),1))]
    return self.activationFunction(X_with_bias, self.weights)

  def total_loss(self):
    total_loss = np.sum(self.error)
    print(f"total loss : \n {total_loss}")
    return total_loss



