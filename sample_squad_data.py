#!/usr/bin/env python
# coding: utf-8



import numpy as np
from datasets import load_dataset
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag, ne_chunk
# nltk.download('averaged_perceptron_tagger')
from spacy.symbols import ORTH
import spacy
import gensim
from nltk.corpus import wordnet
import pickle
from sample_another_data1 import *
# In[9]:

SPECIAL_TOKENS = {"pad": "<pad>", "oov": "<oov>", "sos": "<sos>", "eos": "<eos>"}
SPECIAL_TOKEN2ID = {"<pad>": 0, "<oov>": 1, "<sos>": 2, "<eos>": 3}
#GLOVE = gensim.models.KeyedVectors.load_word2vec_format('glove.840B.300d.txt', binary=False, no_header=True)
with open('glove_model.pkl', 'rb') as f:
    GLOVE = pickle.load(f)

NLP = spacy.load("en_core_web_sm")
for special_token in SPECIAL_TOKENS.values():
    NLP.tokenizer.add_special_case(special_token, [{ORTH: special_token}])
    

def load_raw_data_from_file(file_path):
    ds = load_dataset(file_path)
    return ds
def words_ids(datas):
    word_ids = []
    for data in datas:
        ans_sent = data['ans_sent']
        word_ids.append(get_word_ids(ans_sent))
    return word_ids
def get_style_ids(question, styles, yes_no_ans):
    style = get_style(question, styles, yes_no_ans)
    style_ids = [styles.index(item) for item in style]
    return style_ids


def get_iob(sentence, answer, answer_start):
    ans_sent_tokens = [token.text for token in NLP(sentence)]
    spans = get_token_char_level_spans(sentence, ans_sent_tokens)
    answer_end = answer_start + len(answer)
    answer_span = []
    for idx, span in enumerate(spans):
        if not (answer_end <= span[0] or answer_start >= span[1]):
            answer_span.append(idx)

    y1_in_sent = answer_span[0]
    y2_in_sent = answer_span[-1]
    iob_tags = []
    for i in range(len(ans_sent_tokens)):
        if i < y1_in_sent:
            iob_tags.append('O')
        elif i == y1_in_sent:
            iob_tags.append('B')
        elif y1_in_sent < i <= y2_in_sent:
            iob_tags.append('I')
        else:
            iob_tags.append('O')
    
    return list(iob_tags)


def get_word_ids(sentence):
    tokens = word_tokenize(sentence)
    word_ids = [GLOVE.key_to_index.get(token, -1) for token in tokens]
    return word_ids



def get_sents_from_passage(passage): # lấy ra các câu từ đoạn
    return sent_tokenize(passage)


def get_sent_contain_answer(passage, answer_start):
    list_sents = get_sents_from_passage(passage)
    indexed_sentences = []
    current_index = 0
    new = []
    for sentence in list_sents:
        indexed_sentence = [(char, current_index + i) for i, char in enumerate(sentence)]
        indexed_sentences.append(indexed_sentence)
        current_index += len(sentence) + 1 
    for i, sentence in enumerate(indexed_sentences):
        for char, idx in sentence:
            if answer_start[0] == idx:
                return list_sents[i]
            


def get_synonyms(word):

    synonyms = []
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.append(l.name())
    return list(set(synonyms))

def get_antonyms(word):
    antonyms = []
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            if l.antonyms():
                antonyms.append(l.antonyms()[0].name())
    return list(set(antonyms))

def get_semantic_related_words(token, topN):

    semantic_related = []
    if token in SPECIAL_TOKENS.values():
        semantic_related = [token]
        return semantic_related

    if token in GLOVE.key_to_index:
        token_in_glove = token
    elif token.lower() in GLOVE.key_to_index:
        token_in_glove = token.lower()
    else:
        token_in_glove = None
    if token_in_glove is not None:
        semantic_related = GLOVE.most_similar(positive=[token_in_glove], topn=topN)
        semantic_related = [item[0] for item in semantic_related]

    return semantic_related




def _dfs(doc, doc_token_list, cur_id, cur_path, max_depth, related): # use in get all relate
    if len(cur_path) > max_depth:
        return
    if cur_id in related and len(related[cur_id]) <= len(cur_path):
        return
    related[cur_id] = cur_path
    for token in doc_token_list:
        if token.i != cur_id:
            continue
        new_path = copy.deepcopy(cur_path)
        try:
            new_path.append(token.dep_)
        except:
            continue
        _dfs(doc, doc_token_list, token.head.i, new_path, max_depth, related)
        for child in token.children:
            new_path = copy.deepcopy(cur_path)
            new_path.append(child.head.dep_)
            _dfs(doc, doc_token_list, child.i, new_path, max_depth, related)



            
            
def get_style(questions, styles, yes_no_ans):
    stl_from_data = []
    for question in questions:
        if question.split()[0].lower() in styles:
            stl_from_data.append(question.split()[0].lower())
        elif question.split()[0].lower() in yes_no_ans:
            stl_from_data.append('yes-no')
        else:
            stl_from_data.append('other')   
    return stl_from_data










def get_all_related(context_doc, doc_token_list):
    idx2token = {}
    idx2related = {}
    tokens = []
    for token in context_doc:
        idx2token[token.i] = token
        related = {}
        _dfs(context_doc, doc_token_list, token.i, [], len(context_doc) - 1, related)
        sort_related = sorted(related.items(), key=lambda x: len(x[1]))
        idx2related[token.i] = sort_related
        tokens.append(token.text)
    return idx2token, idx2related, tokens







