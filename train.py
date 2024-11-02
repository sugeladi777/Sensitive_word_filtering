import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
import numpy as np
import jieba
import joblib


# 定义数组模块，默认使用numpy
xp = np


# 加载文本数据集
def load_text_dataset(filename):
    """
    加载文本数据集

    参数:
    filename (str): 数据集文件路径

    返回:
    X (pd.Series): 文本数据
    y (np.ndarray): 标签数据
    """
    data = pd.read_csv(filename, sep='\t', header=None,
                       names=['label', 'text'])
    X = data['text']
    y = xp.array(data['label'] == '__label__0') * 2 - 1
    return X, y


# 加载停用词
def load_stop_words(filepath):
    """
    加载停用词

    参数:
    filepath (str): 停用词文件路径

    返回:
    stop_words (list): 停用词列表
    """
    with open(filepath, 'r', encoding='utf-8') as file:
        stop_words = file.read().splitlines()
    return stop_words


# 将文本数据向量化
def vectorize(train, val, stop_words):
    """
    将文本数据向量化

    参数:
    train (pd.Series): 训练集文本数据
    val (pd.Series): 验证集文本数据
    stop_words (list): 停用词列表

    返回:
    train_normalized (np.ndarray): 训练集向量化数据
    val_normalized (np.ndarray): 验证集向量化数据
    """
    tfidf = TfidfVectorizer(
        tokenizer=jieba.cut, stop_words=stop_words, smooth_idf=True, sublinear_tf=True)
    train_normalized = tfidf.fit_transform(train).toarray()
    val_normalized = tfidf.transform(val).toarray()
    return train_normalized, val_normalized


# 计算准确率
def compute_accuracy(X, y, theta):
    """
    计算准确率

    参数:
    X (np.ndarray): 特征数据
    y (np.ndarray): 标签数据
    theta (np.ndarray): 模型参数

    返回:
    accuracy (float): 准确率
    """
    predictions = xp.sign(xp.dot(X, theta))
    accuracy = xp.mean(predictions == y)
    return accuracy


# 计算RBF核函数
def rbf_kernel(X1, X2, gamma=1):
    """
    计算RBF核函数

    参数:
    X1 (np.ndarray): 特征数据1
    X2 (np.ndarray): 特征数据2
    gamma (float): 核函数参数

    返回:
    K (np.ndarray): 核矩阵
    """
    X1_sq = xp.sum(X1 ** 2, axis=1).reshape(-1, 1)
    X2_sq = xp.sum(X2 ** 2, axis=1).reshape(1, -1)
    K = xp.exp(-gamma * (X1_sq + X2_sq - 2 * xp.dot(X1, X2.T)))
    return K


# 使用子梯度下降法训练核SVM
def kernel_svm_subgrad_descent(X, y, X_val, y_val, alpha=0.1, lambda_reg=1, num_iter=20000, batch_size=10, gamma=1):
    """
    使用子梯度下降法训练核SVM

    参数:
    X (np.ndarray): 训练集特征数据
    y (np.ndarray): 训练集标签数据
    X_val (np.ndarray): 验证集特征数据
    y_val (np.ndarray): 验证集标签数据
    alpha (float): 学习率
    lambda_reg (float): 正则化参数
    num_iter (int): 迭代次数
    batch_size (int): 批量大小
    gamma (float): 核函数参数

    返回:
    theta_hist (np.ndarray): 参数历史记录
    loss_hist (np.ndarray): 损失历史记录
    accuracy_trains (np.ndarray): 训练集准确率历史记录
    accuracy_vals (np.ndarray): 验证集准确率历史记录
    """
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
            print(f"iter {t} loss: {loss} train accuracy: {
                  accuracy_train} val accuracy: {accuracy_val}")

    return theta_hist, loss_hist, accuracy_trains, accuracy_vals


# 绘制损失和准确率曲线图
def plot_loss_accuracy(loss_hist, accuracy_trains, accuracy_vals, alpha, lambda_reg, batch_size, gamma, output_dir="./chart/"):
    """
    绘制损失和准确率曲线图

    参数:
    loss_hist (np.ndarray): 损失历史记录
    accuracy_trains (np.ndarray): 训练集准确率历史记录
    accuracy_vals (np.ndarray): 验证集准确率历史记录
    alpha (float): 学习率
    lambda_reg (float): 正则化参数
    batch_size (int): 批量大小
    gamma (float): 核函数参数
    output_dir (str): 输出目录
    """
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
    plt.savefig(f"{output_dir}gamma{gamma}alpha{alpha}lambda_reg{
                lambda_reg}batch_size{batch_size}.png")
    plt.close()


# 网格搜索以找到最佳参数
def grid_search(X_train_vect, y_train, X_val_vect, y_val, param_grid):
    """
    网格搜索以找到最佳参数

    参数:
    X_train_vect (np.ndarray): 训练集向量化数据
    y_train (np.ndarray): 训练集标签数据
    X_val_vect (np.ndarray): 验证集向量化数据
    y_val (np.ndarray): 验证集标签数据
    param_grid (dict): 参数网格

    返回:
    best_params (dict): 最佳参数
    best_val_accuracy (float): 最佳验证集准确率
    """
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


# 预测函数
def predict(X, theta):
    """
    预测函数

    参数:
    X (np.ndarray): 特征数据
    theta (np.ndarray): 模型参数

    返回:
    predictions (np.ndarray): 预测结果
    """
    predictions = xp.sign(xp.dot(X, theta))
    return predictions


# 主函数
def main():
    """
    主函数
    """
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
