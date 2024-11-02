# Sensitive_word_filtering

## 简介

本项目实现了一个敏感词过滤系统，使用Aho-Corasick自动机进行高效的敏感词匹配，其次还使用支持向量机对文本进行预测分类。

## 文件说明

- `ac.py`: 构建Aho-Corasick自动机并进行敏感词过滤的主程序。
- `train.py`: 训练核SVM模型的代码。
- `predict.py`: 使用训练好的模型进行预测的代码。
- `requirements.txt`: 项目依赖的Python库列表。
- `model_params.pkl`：提前训练好的一个模型，可以直接使用来进行预测。
- `data/`: 存放训练和测试数据的文件夹。
  - `train.txt`: 训练数据集。
  - `test.txt`: 测试数据集。
- `stopwords/`: 存放停用词的文件夹。
  - `hit_stopwords.txt`: 停用词列表。
- `dicts/`: 存放敏感词词典的文件夹。
  - `illegal.txt`: 敏感词词典。

## 安装依赖

在运行本项目之前，请确保已安装所有依赖库。你可以使用以下命令安装依赖：

```bash
pip install -r requirements.txt
```

## 数据集来源

- `data`中的数据来源于`https://github.com/wjx-git/IllegalTextDetection.git`
- `dicts`中的数据来源于`https://github.com/wjx-git/IllegalTextDetection.git`
- `stopwords`中的数据来源于`https://github.com/goto456/stopwords.git`
