import ahocorasick
import opencc


# 构建Aho-Corasick自动机
def build_actree(wordlist):
    """
    构建Aho-Corasick自动机

    参数:
    wordlist (list): 词汇列表

    返回:
    actree (ahocorasick.Automaton): 构建好的Aho-Corasick自动机
    """
    actree = ahocorasick.Automaton()
    for index, word in enumerate(wordlist):
        actree.add_word(word, (index, word))
    actree.make_automaton()
    return actree


# 从文件中读取词汇列表
def read_wordlist_from_file(filepath):
    """
    从文件中读取词汇列表

    参数:
    filepath (str): 文件路径

    返回:
    wordlist (list): 词汇列表
    """
    with open(filepath, 'r', encoding='utf-8') as file:
        wordlist = [line.strip() for line in file.readlines()]
    return wordlist


if __name__ == '__main__':
    # 初始化OpenCC转换器，将繁体转换为简体
    converter = opencc.OpenCC('t2s')

    # 从文件中读取词汇列表并转换为简体
    wordlist = read_wordlist_from_file('./dicts/illegal.txt')
    wordlist = [converter.convert(word) for word in wordlist]

    # 构建Aho-Corasick自动机
    actree = build_actree(wordlist=wordlist)

    # 输入文本并转换为简体
    sent = input("请输入一段文本：")
    sent = converter.convert(sent)
    sent_cp = sent

    # 遍历文本中的敏感词并进行屏蔽
    for i in actree.iter(sent):
        sent_cp = sent_cp.replace(i[1][1], "**")
        print("屏蔽词：", i[1][1])

    # 输出屏蔽结果
    print("屏蔽结果：", sent_cp)
