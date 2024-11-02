import ahocorasick
import opencc


def build_actree(wordlist):
    actree = ahocorasick.Automaton()
    for index, word in enumerate(wordlist):
        actree.add_word(word, (index, word))
    actree.make_automaton()
    return actree


def read_wordlist_from_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        wordlist = [line.strip() for line in file.readlines()]
    return wordlist


if __name__ == '__main__':
    converter = opencc.OpenCC('t2s')
    wordlist = read_wordlist_from_file('./dicts/illegal.txt')
    wordlist = [converter.convert(word) for word in wordlist]
    actree = build_actree(wordlist=wordlist)
    sent = input("请输入一段文本：")
    sent = converter.convert(sent)
    sent_cp = sent
    for i in actree.iter(sent):
        sent_cp = sent_cp.replace(i[1][1], "**")
        print("屏蔽词：", i[1][1])
    print("屏蔽结果：", sent_cp)
