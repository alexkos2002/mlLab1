
import pandas as pd
import numpy as nmp

learning_rate = 0.8
tolerance = 0.0001

class LinearRegression:
    def __init__(self,  learning_rate, tolerance):
        self.lr = learning_rate
        self.tol = tolerance
        self.n_features = 0
        self.n_samples = 0


    def prediction(self, X, w):
        return nmp.dot(w.T, X)

    def loss(self, Y, prediction):
        L = 1 / self.n_samples * nmp.sum(nmp.power(Y - prediction, 2))
        return L

    def gradientDescent(self, w, X, Y, prediction):
        dldW = 2 / self.n_samples * nmp.dot(X, (prediction - Y).T)
        w = w - self.lr * dldW
        return w

    def trainViaLinearRegression(self, X, Y):
        x1 = nmp.ones((1, X.shape[1]))
        X = nmp.append(X, x1, 0)

        self.n_features = X.shape[0]
        self.n_samples = X.shape[1]

        w = nmp.zeros((self.n_features, 1))

        curTolerance = float('inf')
        prevLoss = float('inf')
        iteration = 0

        print("Starting training...")
        while curTolerance > self.tol:
            prediction = self.prediction(X, w)
            loss = self.loss(Y, prediction)

            w = self.gradientDescent(w, X, Y, prediction)

            curTolerance = prevLoss - loss

            print("Iteration №" + str(iteration))
            print("curTolerance: " + str(curTolerance))

            prevLoss = loss
            iteration += 1

        print("Result weights: " + str(w))
        print("Training was finished.")

        return w

    def testLinearRegression(self, X, w):
        x1 = nmp.ones((1, X.shape[1]))
        X = nmp.append(X, x1, 0)
        return self.prediction(X, w)

    # the metric for estimating the regression quality is a sum of
    # squared residuals for test set divided by the number of samples
    # It is calculated similar to loss function
    def estimateRegressionQuality(self, Y_result, Y_training):
        return nmp.sum(nmp.power(Y_training - Y_result, 2)) / Y_training.shape[1]

if __name__ == '__main__':

    training_df = pd.read_csv("lab_1_train.csv")
    testing_df = pd.read_csv("lab_1_test.csv")
    print(training_df)
    print("\n")
    X_training = (training_df['x']).to_numpy()
    Y_training = (training_df['y']).to_numpy()
    X_training = nmp.reshape(X_training, (1, X_training.shape[0]))
    Y_training = nmp.reshape(Y_training, (1, Y_training.shape[0]))

    X_testing = (testing_df['x']).to_numpy()
    X_testing = nmp.reshape(X_testing, (1, X_testing.shape[0]))
    Y_testing = (testing_df['y']).to_numpy()
    Y_testing = nmp.reshape(Y_testing, (1, Y_testing.shape[0]))

    linearRegression = LinearRegression(learning_rate, tolerance)

    w = linearRegression.trainViaLinearRegression(X_training, Y_training)

    Y_result = linearRegression.testLinearRegression(X_testing, w)

    print("\n")
    print("Results on testing set: ")
    print(Y_result)

    print("The quality of regression is " + str(linearRegression.estimateRegressionQuality(Y_result, Y_testing)))

    # The best learning rate from the regression quality perspective is between 0.8 and 0.9


