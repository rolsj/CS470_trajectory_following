"""
Regression Model.

Usage:
    import numpy as np
    
    x_train = np.array(
        [[5, 3, 10], [4, 4, 20], [2, 2, 1], [4, 5, 10], [3, 3, 20], [2, 4, 10], [10, 8, 6], [8, 10, 10], [1, 20, 20], [20, 1, 1]]
    )

    y_train = np.array(
        [120, 240, 10, 185, 210, 125, 160, 240, 300, 280]
    )

    model = Regression()   # Set `verbose = True` to print log to console
    model.train_with(x_train, y_train)

    x_predict = [4, 8, 5]
    y_predict = model.predict(x_predict)
    print(f"{x_predict=}, {y_predict=}")

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


class Regression:

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.lr = None

    def train_with(self, x_train, y_train):
        self.lr = LinearRegression()
        self.lr.fit(x_train, y_train)
        if self.verbose:
            print("[Regression] train finished")
    
    def predict(self, x_predict):
        y_predict = self.lr.predict(x_predict)
        if self.verbose:
            print(f"[Regression] finish predicting {len(y_predict)} data")
        return y_predict

    # Warning: you need to set DISPLAG
    def show_3d_plot(self, x, y):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(projection="3d")
        ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=y, cmap="coolwarm")
        plt.show()



# For Test
if __name__ == "__main__":
    x_train = np.array(
        [[5, 3, 10], [4, 4, 20], [2, 2, 1], [4, 5, 10], [3, 3, 20], [2, 4, 10], [10, 8, 6], [8, 10, 10], [1, 20, 20], [20, 1, 1]]
    )

    y_train = np.array(
        [120, 240, 10, 185, 210, 125, 160, 240, 300, 280]
    )

    model = Regression(verbose=True)
    model.train_with(x_train, y_train)

    predict_cnt = 1000

    x_predict = []
    for it in range(predict_cnt):
        h1 = np.random.uniform(1, 20)
        h2 = np.random.uniform(1, 20)
        l = np.random.uniform(1, 20)
        x_predict.append([h1, h2, l])
    x_predict = np.array(x_predict)
    print(f"{len(x_predict)} input data generated")
    y_predict = model.predict(x_predict)
    print(f"{len(y_predict)} outputs predicted")

    for i in range(10):
        print((x_predict[i], y_predict[i]))
    model.show_3d_plot(x_predict, y_predict)

