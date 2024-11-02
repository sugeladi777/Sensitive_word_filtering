import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
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
def vectorize(train, stop_words):
    """
    将文本数据向量化

    参数:
    train (pd.Series): 训练集文本数据
    stop_words (list): 停用词列表

    返回:
    train_normalized (np.ndarray): 训练集向量化数据
    tfidf (TfidfVectorizer): 训练好的TfidfVectorizer对象
    """
    tfidf = TfidfVectorizer(
        tokenizer=jieba.cut, stop_words=stop_words, smooth_idf=True, sublinear_tf=True)
    train_normalized = tfidf.fit_transform(train).toarray()
    return train_normalized, tfidf


# 计算RBF核函数
def rbf_kernel(X1_sq, X1, X2, gamma=1):
    """
    计算RBF核函数

    参数:
    X1_sq (np.ndarray): 特征数据1的平方和
    X1 (np.ndarray): 特征数据1
    X2 (np.ndarray): 特征数据2
    gamma (float): 核函数参数

    返回:
    K (np.ndarray): 核矩阵
    """
    X2_sq = xp.sum(X2 ** 2, axis=1).reshape(1, -1)
    K = xp.exp(-gamma * (X1_sq + X2_sq - 2 * xp.dot(X1, X2.T)))
    return K


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


# 加载模型参数
def load_model(model_path):
    """
    加载模型参数

    参数:
    model_path (str): 模型文件路径

    返回:
    best_params (dict): 最佳参数
    """
    model_data = joblib.load(model_path)
    best_params = model_data['best_params']
    return best_params


# 主函数
def main():
    """
    主函数
    """
    # 加载模型参数
    best_params = load_model('model_params.pkl')
    theta = best_params['theta']
    gamma = best_params['gamma']

    # 加载停用词
    stop_words = load_stop_words("./stopwords/hit_stopwords.txt")

    # 加载训练数据以获取相同的 TfidfVectorizer
    X_train, _ = load_text_dataset("./data/train.txt")
    _, tfidf = vectorize(X_train, stop_words)
    # 将训练数据转成向量表示
    X_train_vect = tfidf.transform(X_train).toarray()
    X_train_vect = np.hstack(
        (X_train_vect, np.ones((X_train_vect.shape[0], 1))))  # 增加偏置项
    X1_sq = xp.sum(X_train_vect ** 2, axis=1).reshape(-1, 1)

    while True:
        # 输入文本
        input_text = input("请输入文本: ")
        # 将输入文本转成向量表示
        input_vect = tfidf.transform([input_text]).toarray()
        input_vect = np.hstack(
            (input_vect, np.ones((input_vect.shape[0], 1))))  # 增加偏置项

        # 计算核矩阵
        K_input = rbf_kernel(X1_sq=X1_sq, X1=X_train_vect,
                             X2=input_vect, gamma=gamma)

        # 进行预测
        predictions = predict(K_input.T, theta)
        print(f"Predictions: {predictions}")


if __name__ == '__main__':
    main()
