import pandas as pd
# import cupy as cp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
import numpy as np
import jieba
import joblib


# def get_array_module():
#     try:
#         # 尝试使用cupy
#         cp.cuda.runtime.getDeviceCount()
#         return cp
#     except cp.cuda.runtime.CUDARuntimeError:
#         # 如果GPU不可用，则使用numpy
#         return np


xp = np


def load_text_dataset(filename):
    data = pd.read_csv(filename, sep='\t', header=None,
                       names=['label', 'text'])
    X = data['text']
    y = xp.array(data['label'] == '__label__0') * 2 - 1
    return X, y


def load_stop_words(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        stop_words = file.read().splitlines()
    return stop_words


def vectorize(train, val, stop_words):
    tfidf = TfidfVectorizer(tokenizer=jieba.cut, stop_words=stop_words,
                            smooth_idf=True, sublinear_tf=True)
    train_normalized = tfidf.fit_transform(train).toarray()
    val_normalized = tfidf.transform(val).toarray()
    return train_normalized, val_normalized


def compute_accuracy(X, y, theta):
    predictions = xp.sign(xp.dot(X, theta))
    accuracy = xp.mean(predictions == y)
    return accuracy


def rbf_kernel(X1, X2, gamma=1):
    X1_sq = xp.sum(X1 ** 2, axis=1).reshape(-1, 1)
    X2_sq = xp.sum(X2 ** 2, axis=1).reshape(1, -1)
    K = xp.exp(-gamma * (X1_sq + X2_sq - 2 * xp.dot(X1, X2.T)))
    return K


def kernel_svm_subgrad_descent(X, y, X_val, y_val, alpha=0.1, lambda_reg=1, num_iter=20000, batch_size=10, gamma=1):
    num_instances = X.shape[0]
    theta = xp.ones(num_instances)
    theta_hist = xp.zeros((num_iter + 1, num_instances))
    loss_hist = xp.zeros((num_iter,))
    accuracy_trains = xp.zeros(num_iter)
    accuracy_vals = xp.zeros(num_iter)

    K = rbf_kernel(X, X, gamma=gamma)
    K_val = rbf_kernel(X, X_val, gamma=gamma)

    for t in range(num_iter):
        batch_indices = xp.random.choice(
            num_instances, batch_size, replace=False)
        K_batch = K[batch_indices]
        y_batch = y[batch_indices]
        margin = y_batch * xp.dot(K_batch, theta)
        indicator = margin < 1
        subgrad = lambda_reg * theta - \
            xp.dot(K_batch.T, y_batch * indicator) / batch_size
        theta -= alpha * subgrad
        hinge_loss = xp.mean(xp.maximum(0, 1 - y * xp.dot(K, theta)))
        loss = lambda_reg * xp.linalg.norm(theta) ** 2 + hinge_loss
        accuracy_train = compute_accuracy(K_batch, y_batch, theta)
        accuracy_val = compute_accuracy(K_val.T, y_val, theta)
        theta_hist[t + 1] = theta
        loss_hist[t] = loss
        accuracy_trains[t] = accuracy_train
        accuracy_vals[t] = accuracy_val
        if t % 100 == 0:
            print(
                f"iter {t} loss: {loss} train accuracy: {accuracy_train} val accuracy: {accuracy_val}")

    return theta_hist, loss_hist, accuracy_trains, accuracy_vals


def plot_loss_accuracy(loss_hist, accuracy_trains, accuracy_vals, alpha, lambda_reg, batch_size, gamma, output_dir="./chart/"):
    plt.figure(figsize=(12, 6))
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(loss_hist, label='Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss over Iterations')
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(accuracy_trains, label='Training Accuracy')
    plt.plot(accuracy_vals, label='Validation Accuracy')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Iterations')
    plt.legend()

    # 添加训练信息
    train_info = (
        f"alpha: {alpha}\n"
        f"lambda_reg: {lambda_reg}\n"
        f"batch_size: {batch_size}\n"
        f"gamma: {gamma}\n"
        f"Final Training Accuracy: {accuracy_trains[-1]:.4f}\n"
        f"Final Validation Accuracy: {accuracy_vals[-1]:.4f}"
    )
    plt.gcf().text(0.5, 0.01, train_info, ha='center',
                   fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    plt.tight_layout()
    # 保存图表到指定目录
    plt.savefig(
        f"{output_dir}gamma{gamma}alpha{alpha}lambda_reg{lambda_reg}batch_size{batch_size}.png")
    plt.close()


def grid_search(X_train_vect, y_train, X_val_vect, y_val, param_grid):
    best_params = None
    best_val_accuracy = -np.inf

    for params in ParameterGrid(param_grid):
        lambda_reg = params['lambda_reg']
        batch_size = params['batch_size']
        gamma = params['gamma']
        alpha = params['alpha']

        # 训练模型
        theta_hist, loss_hist, accuracy_trains, accuracy_vals = kernel_svm_subgrad_descent(
            X_train_vect, y_train, X_val_vect, y_val, alpha=alpha, lambda_reg=lambda_reg, num_iter=100000, batch_size=batch_size, gamma=gamma)

        # 获取最后一个验证准确率
        final_val_accuracy = accuracy_vals[-1]

        # 更新最佳参数
        if final_val_accuracy > best_val_accuracy:
            best_val_accuracy = final_val_accuracy
            best_params = params
            best_params['theta'] = theta_hist[-1]

        # 绘制损失和准确率曲线图
        plot_loss_accuracy(loss_hist, accuracy_trains,
                           accuracy_vals, alpha, lambda_reg, batch_size, gamma)

    return best_params, best_val_accuracy


def predict(X, theta):
    predictions = xp.sign(xp.dot(X, theta))
    return predictions


def main():
    X_train, y_train = load_text_dataset("./data/train.txt")
    X_val, y_val = load_text_dataset("./data/test.txt")
    stop_words = load_stop_words("./stopwords/hit_stopwords.txt")

    # 将训练集和验证集中的文本转成向量表示
    X_train_vect, X_val_vect = vectorize(X_train, X_val, stop_words)
    X_train_vect = xp.hstack(
        (X_train_vect, xp.ones((X_train_vect.shape[0], 1))))  # 增加偏置项
    X_val_vect = xp.hstack(
        (X_val_vect, xp.ones((X_val_vect.shape[0], 1))))  # 增加偏置项

    # 将数据转移到GPU（如果可用）
    X_train_vect = xp.asarray(X_train_vect)
    y_train = xp.asarray(y_train)
    X_val_vect = xp.asarray(X_val_vect)
    y_val = xp.asarray(y_val)

    # 定义参数网格
    param_grid = {
        'alpha': [0.05],
        'lambda_reg': [1e-05],
        'batch_size': [1000],
        'gamma': [2]
    }

    # 执行网格搜索
    best_params, best_val_accuracy = grid_search(
        X_train_vect, y_train, X_val_vect, y_val, param_grid)
    print(f"Best Validation Accuracy: {best_val_accuracy}")
    print(f"Best Parameters: {best_params}")

    # 保存最佳模型参数
    joblib.dump({'best_params': best_params}, 'model_params.pkl')


if __name__ == '__main__':
    main()
