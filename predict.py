import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import jieba
import joblib

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


def vectorize(train, stop_words):
    tfidf = TfidfVectorizer(tokenizer=jieba.cut, stop_words=stop_words,
                            smooth_idf=True, sublinear_tf=True)
    train_normalized = tfidf.fit_transform(train).toarray()
    return train_normalized, tfidf


def rbf_kernel(X1_sq, X1, X2, gamma=1):
    X2_sq = xp.sum(X2 ** 2, axis=1).reshape(1, -1)
    K = xp.exp(-gamma * (X1_sq + X2_sq - 2 * xp.dot(X1, X2.T)))
    return K


def predict(X, theta):
    predictions = xp.sign(xp.dot(X, theta))
    return predictions


def load_model(model_path):
    model_data = joblib.load(model_path)
    best_params = model_data['best_params']
    return best_params


def main():
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
