# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief: corrector with spell and stroke

import codecs
import os

from pypinyin import lazy_pinyin

from pycorrector.detector import detect
from pycorrector.detector import get_frequency
from pycorrector.detector import get_ppl_score
from pycorrector.detector import trigram_char
from pycorrector.detector import word_freq
from pycorrector.text_preprocess import is_chinese_string
from pycorrector.util import dump_pkl
from pycorrector.util import load_pkl

pwd_path = os.path.abspath(os.path.dirname(__file__))
char_file_path = os.path.join(pwd_path, 'data/cn/char_set.txt')


def load_word_dict(path):
    word_dict = ''
    with codecs.open(path, 'r', encoding='utf-8') as f:
        for w in f:
            word_dict += w.strip()
    return word_dict


def load_same_pinyin(path, sep='\t'):
    """
    加载同音字
    :param path:
    :return:
    """
    result = dict()
    if not os.path.exists(path):
        print("file not exists:", path)
        return result
    with codecs.open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(sep)
            if parts and len(parts) > 2:
                key_char = parts[0]
                same_pron_same_tone = set(list(parts[1]))
                same_pron_diff_tone = set(list(parts[2]))
                value = same_pron_same_tone.union(same_pron_diff_tone)
                if len(key_char) > 1 or not value:
                    continue
                result[key_char] = value
    return result


def load_same_stroke(path, sep='\t'):
    """
    加载形似字
    :param path:
    :param sep:
    :return:
    """
    result = dict()
    if not os.path.exists(path):
        print("file not exists:", path)
        return result
    with codecs.open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(sep)
            if parts and len(parts) > 1:
                key_char = parts[0]
                result[key_char] = set(list(parts[1]))
    return result


cn_char_set = load_word_dict(char_file_path)
same_pinyin_text_path = os.path.join(pwd_path, 'data/same_pinyin.txt')
same_pinyin_model_path = os.path.join(pwd_path, 'data/same_pinyin.pkl')
# 同音字
if os.path.exists(same_pinyin_model_path):
    same_pinyin = load_pkl(same_pinyin_model_path)
else:
    print('load same pinyin from text file:', same_pinyin_text_path)
    same_pinyin = load_same_pinyin(same_pinyin_text_path)
    dump_pkl(same_pinyin, same_pinyin_model_path)

# 形似字
same_stroke_text_path = os.path.join(pwd_path, 'data/same_stroke.txt')
same_stroke_model_path = os.path.join(pwd_path, 'data/same_stroke.pkl')
if os.path.exists(same_stroke_model_path):
    same_stroke = load_pkl(same_stroke_model_path)
else:
    print('load same stroke from text file:', same_stroke_text_path)
    same_stroke = load_same_stroke(same_stroke_text_path)
    dump_pkl(same_stroke, same_stroke_model_path)


def get_same_pinyin(char):
    """
    取同音字
    :param char:
    :return:
    """
    return same_pinyin.get(char, set())


def get_same_stroke(char):
    """
    取形似字
    :param char:
    :return:
    """
    return same_stroke.get(char, set())


def _edit_distance_word(word, char_set):
    """
    all edits that are one edit away from 'word'
    :param word:
    :param char_set:
    :return:
    """
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in char_set]
    return set(transposes + replaces)


def _known(words):
    return set(word for word in words if word in word_freq)


def _get_confusion_set(c):
    confusion_char_set = get_same_pinyin(c)
    if not confusion_char_set:
        confusion_char_set = set()
    return confusion_char_set


def _generate_items(word, fraction=2):
    if not is_chinese_string(word):
        return []
    candidates_1_order = []
    candidates_2_order = []
    candidates_3_order = []
    candidate_words = list(_known(_edit_distance_word(word, cn_char_set)))
    for candidate_word in candidate_words:
        if lazy_pinyin(candidate_word) == lazy_pinyin(word):
            # same pinyin
            candidates_1_order.append(candidate_word)
    if len(word) == 1:
        # same pinyin
        confusion_char_set = _get_confusion_set(word[0])
        confusion = [i for i in confusion_char_set if i]
        candidates_2_order.extend(confusion)
    if len(word) > 1:
        # same first pinyin
        confusion_char_set = _get_confusion_set(word[0])
        confusion = [i + word[1:] for i in confusion_char_set if i]
        candidates_2_order.extend(confusion)
        # same last pinyin
        confusion_char_set = _get_confusion_set(word[-1])
        confusion = [word[:-1] + i for i in confusion_char_set]
        candidates_2_order.extend(confusion)
        if len(word) > 2:
            # same mid pinyin
            confusion_char_set = _get_confusion_set(word[1])
            confusion = [word[0] + i + word[2:] for i in confusion_char_set]
            candidates_3_order.extend(confusion)
    # add all confusion word list
    confusion_word_set = set(candidates_1_order + candidates_2_order + candidates_3_order)
    confusion_word_list = [item for item in confusion_word_set if is_chinese_string(item)]
    confusion_sorted = sorted(confusion_word_list, key=lambda k: \
        get_frequency(k), reverse=True)
    return confusion_sorted[:len(confusion_word_list) // fraction + 1]


def get_sub_array(nums):
    """
    取所有连续子串，
    [0, 1, 2, 5, 7, 8]
    => [[0, 2], 5, [7, 8]]
    :param nums: sorted(list)
    :return:
    """
    ret = []
    for i, c in enumerate(nums):
        if i == 0:
            pass
        elif i <= ii:
            continue
        elif i == len(nums) - 1:
            ret.append([c])
            break
        ii = i
        cc = c
        # get continuity Substring
        while ii < len(nums) - 1 and nums[ii + 1] == cc + 1:
            ii = ii + 1
            cc = cc + 1
        if ii > i:
            ret.append([c, nums[ii] + 1])
        else:
            ret.append([c])
    return ret


def _correct_item(sentence, idx, item):
    """
    纠正错误，逐词处理
    :param sentence:
    :param idx:
    :param item:
    :return: corrected word 修正的词语
    """
    corrected_sent = sentence
    # 取得所有可能正确的词
    maybe_error_items = _generate_items(item)
    maybe_error_items = sorted(maybe_error_items, key=lambda k: \
        get_frequency(k), reverse=True)
    if not maybe_error_items:
        return corrected_sent, []
    ids = idx.split(',')
    begin_id = int(ids[0])
    end_id = int(ids[-1]) if len(ids) > 1 else int(ids[0]) + 1
    before = sentence[:begin_id]
    after = sentence[end_id:]
    corrected_item = min(maybe_error_items,
                         key=lambda k: get_ppl_score(list(before + k + after),
                                                     mode=trigram_char))
    wrongs, rights, begin_idx, end_idx = [], [], [], []
    if corrected_item != item:
        corrected_sent = before + corrected_item + after
        print('pred:', item, '=>', corrected_item)
        wrongs.append(item)
        rights.append(corrected_item)
        begin_idx.append(begin_id)
        end_idx.append(end_id)
    detail = list(zip(wrongs, rights, begin_idx, end_idx))
    return corrected_sent, detail


def correct(sentence):
    """
    句子改错
    :param sentence: 句子文本
    :return: 改正后的句子, list(wrongs, rights, begin_idx, end_idx)
    """
    detail = []
    maybe_error_ids = get_sub_array(detect(sentence))
    # print('maybe_error_ids:', maybe_error_ids)
    # 取得字词对应表
    index_char_dict = dict()
    for index in maybe_error_ids:
        if len(index) == 1:
            # 字
            index_char_dict[','.join(map(str, index))] = sentence[index[0]]
        else:
            # 词
            index_char_dict[','.join(map(str, index))] = sentence[index[0]:index[-1]]
    for index, item in index_char_dict.items():
        # 字词纠错
        sentence, detail_word = _correct_item(sentence, index, item)
        if detail_word:
            detail.append(detail_word)
    return sentence, detail


if __name__ == '__main__':
    line = '少先队员因该为老人让坐'
    line = '机七学习是人工智能领遇最能体现智能的一个分知'
    print('input sentence is:', line)
    corrected_sent, detail = correct(line)
    print('corrected_sent:', corrected_sent)
    print('detail:', detail)
