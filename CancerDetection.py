import numpy as np
import matplotlib.pyplot as plt
import copy
import math


class LogisticRegression:
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_cost_logistic(self, X, y, w, b):
        m = X.shape[0]
        z = np.dot(X, w) + b
        f_wb = self.sigmoid(z)
        cost = -y * np.log(f_wb) - (1 - y) * np.log(1 - f_wb)
        total_cost = np.sum(cost) / m
        return total_cost

    def compute_gradient_descent(self, X, y, w, b):
        m = X.shape[0]
        f_wb = self.sigmoid(np.dot(X, w) + b)
        err = f_wb - y
        dj_dw = np.dot(X.T, err) / m
        dj_db = np.sum(err) / m
        return dj_db, dj_dw

    def gradient_descent(self, X, y, w_in, b_in, alpha, num_iters):
        J_history = []
        w = copy.deepcopy(w_in)
        b = b_in

        for i in range(num_iters):
            dj_db, dj_dw = self.compute_gradient_descent(X, y, w, b)
            w = w - alpha * dj_dw
            b = b - alpha * dj_db

            if i % math.ceil(num_iters / 10) == 0:
                cost = self.compute_cost_logistic(X, y, w, b)
                J_history.append(cost)

        return w, b, J_history

    def predict(self, X, w, b):
        lr = LogisticRegression()
        m, n = X.shape
        predict = np.zeros((m,))
        for i in range(m):
            z_wb = 0
            for j in range(n):
                z_wb_ij = X[i, j] * w[j]
                z_wb += z_wb_ij
            z_wb += b
            f_wb = lr.sigmoid(z_wb)
            predict[i] = f_wb >= 0.5
        return predict


class RegularizedLogisticRegression:

    def compute_cost_logistic_reg(X, y, w, b, lambda_=1):
        lr = LogisticRegression()
        m, n = X.shape
        cost = 0.
        for i in range(m):
            z_i = np.dot(X[i], w) + b
            f_wb_i = lr.sigmoid(z_i)
            cost += -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)

        cost = cost/m

        reg_cost = 0
        for j in range(n):
            reg_cost += (w[j]**2)
        reg_cost = (lambda_/(2*m)) * reg_cost

        total_cost = cost + reg_cost
        return total_cost

    def compute_gradient_logistic_reg(X, y, w, b, lambda_):
        lr = LogisticRegression()
        m, n = X.shape
        dj_dw = np.zeros((n,))
        dj_db = 0.0

        for i in range(m):
            f_wb_i = lr.sigmoid(np.dot(X[i], w) + b)
            err_i = f_wb_i - y[i]
            for j in range(n):
                dj_dw[j] = dj_dw[j] + err_i * X[i, j]
            dj_db = dj_db + err_i
        dj_dw = dj_dw/m
        dj_db = dj_db/m

        for j in range(n):
            dj_dw[j] = dj_dw[j] + (lambda_/m) * w[j]

        return dj_db, dj_dw


class TrainerandDriver:
    def training_data():
        np.random.seed(42)

        num_samples_per_class = 1000

        mean_class_0 = [1, 1]
        cov_class_0 = [[1, 0.5], [0.5, 1]]
        class_0 = np.random.multivariate_normal(
            mean_class_0, cov_class_0, num_samples_per_class)

        mean_class_1 = [4, 4]
        cov_class_1 = [[1, -0.5], [-0.5, 1]]
        class_1 = np.random.multivariate_normal(
            mean_class_1, cov_class_1, num_samples_per_class)

        X_large = np.vstack([class_0, class_1])
        y_large = np.hstack([np.zeros(num_samples_per_class),
                            np.ones(num_samples_per_class)])

        shuffle_index_large = np.random.permutation(X_large.shape[0])
        X_large_shuffled = X_large[shuffle_index_large]
        y_large_shuffled = y_large[shuffle_index_large]

        split_ratio_large = 0.8
        split_index_large = int(split_ratio_large * X_large.shape[0])

        X_train_large = X_large_shuffled[:split_index_large]
        y_train_large = y_large_shuffled[:split_index_large]

        X_test_large = X_large_shuffled[split_index_large:]
        y_test_large = y_large_shuffled[split_index_large:]

        return X_train_large, y_train_large, X_test_large, y_test_large

    def main(self):
        lr = LogisticRegression()
        X_train, y_train, X_test, y_test = self.training_data()
        w_tmp = np.array([2.0, 3.0])
        b_tmp = 0.0
        alpha = 1.0e-2
        w_final, b_final, J_cost = lr.gradient_descent(
            X_train, y_train, w_tmp, b_tmp, alpha, 1000000)
        print("--------------------CANCER DETECTION--------------------")
        while True:
            choice = input("ENTER 1 TO PREDICT CANCER AND 0 TO EXIT : ")
            if choice == "0":
                break
            elif choice == "1":
                print("ENTER THE DETAILS OF TUMOR")
                size = input("ENTER SIZE OF TUMOR : ")
                node = input("ENTER NUMBER OF NODES : ")
                X_test = np.array([[float(size), float(node)]])
                y_pred = lr.predict(X_test, w_final, b_final)
                if y_pred == 1:
                    print("RESULT : TUMOR IS CANCEROUS")
                else:
                    print("RESULT : TUMOR IS NOT CANCEROUS")
                print("-------------------------------------------------------")
            else:
                print("!INVALID CHOICE!")


if __name__ == "__main__":
    TrainerandDriver.main(TrainerandDriver)
