
import numpy as np
from datasets import load_dataset
from tensorflow.keras.preprocessing.text import Tokenizer
import nltk
import spacy
import benepar
import copy
import math
from nltk.tokenize import sent_tokenize
import re
from datetime import datetime
import os
import sys
import pickle
import codecs
from collections import Counter



def save(filepath, obj, message=None):
    if message is not None:
        print("Saving {}...".format(message))
    pickle_dump_large_file(obj, filepath)


def load(filepath, message=None):
    if message is not None:
        print("Saving {}...".format(message))
    result = pickle_load_large_file(filepath)
    return result


def pickle_dump_large_file(obj, filepath):

    max_bytes = 2**31 - 1
    bytes_out = pickle.dumps(obj)
    n_bytes = sys.getsizeof(bytes_out)
    with open(filepath, 'wb') as f_out:
        for idx in range(0, n_bytes, max_bytes):
            f_out.write(bytes_out[idx:idx + max_bytes])


def pickle_load_large_file(filepath):

    max_bytes = 2**31 - 1
    input_size = os.path.getsize(filepath)
    bytes_in = bytearray(0)
    with open(filepath, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    obj = pickle.loads(bytes_in)
    return obj


NLP = spacy.load("en_core_web_sm")
PARSER = benepar.Parser("benepar_en3")
INFO_QUESTION_TYPES = [
    "Who", "Where", "When", "Why", "Which", "What", "How"]
Q_TYPE2ID_DICT = {
    "What": 0, "Who": 1, "How": 2,
    "Where": 3, "When": 4, "Why": 5,
    "Which": 6, "Boolean": 7, "Other": 8}
NOT_BEGIN_TOKENS_FOR_ANSWER_CLUE = [
    'of', 'for', 'to', 'is', 'are', 'and', 'was', 'were',
    ',', '?', ';', '!', '.']  # TODO: maybe more tokens

f_func_words = open('function_words.txt', "r", encoding="utf-8")
func_words = f_func_words.readlines()
FUNCTION_WORDS_LIST = [word.rstrip() for word in func_words]


def get_squad_raw_examples(filename, debug=False, debug_length=20):
    print("Start get SQuAD raw examples ...")
    start = datetime.now()
    raw_examples = []
    with codecs.open(filename, encoding="utf-8") as fh:
        lines = fh.readlines()
        num_examples = 0
        for line in lines:
            fields = line.strip().split("\t")
            ans_sent = fields[6]
            answer_text = fields[8]
            question = fields[9]

            answer_start_token = int(fields[1].split(" ")[0])
            token_spans = get_token_char_level_spans(
                fields[0], fields[0].split())
            answer_start_in_tokenized_sent = token_spans[answer_start_token][0]

            answer_spans = get_match_spans(answer_text, ans_sent)
            if len(answer_spans) == 0:
                try:
                    print("pattern: ", answer_text)
                    print("match: ", ans_sent)
                except:
                    continue
            answer_start = answer_spans[0][0]
            choice = 0
            gap = abs(answer_start - answer_start_in_tokenized_sent)
            if len(answer_spans) > 1:
                for i in range(len(answer_spans)):
                    new_gap = abs(
                        answer_spans[i][0] - answer_start_in_tokenized_sent)
                    if new_gap < gap:
                        choice = i
                        gap = new_gap
                answer_start = answer_spans[choice][0]

            example = {
                "question": question,
                "ans_sent": ans_sent,
                "answer_text": answer_text,
                "answer_start": answer_start}
            raw_examples.append(example)
            num_examples += 1
            if debug and num_examples >= debug_length:
                break
    print(("Time of get raw examples: {}").format(datetime.now() - start))
    print("Number of raw examples: ", len(raw_examples))
    return raw_examples

def get_token_char_level_spans(text, tokens):

    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def get_match_spans(pattern, input):
    spans = []
    for match in re.finditer(re.escape(pattern), input):
        spans.append(match.span())
    return spans



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

# def get_dataset_info(filename, save_file=None,
#                      sent_limit=100, ques_limit=50, answer_limit=30, is_clue_topN=20,
#                      debug=False, debug_length=20):#GG func 0.1
#     raw_examples = get_squad_raw_examples(filename)# import or tự code
#     examples_with_info = []
#     for i in range(len(raw_examples)):
#         e = raw_examples[i]
#         sentence = e["ans_sent"]
#         question = e["question"]
#         answer = e["answer_text"]
#         answer_start = e["answer_start"]
#         new_e = get_answer_clue_style_info(
#             sentence, question, answer, answer_start,
#             sent_limit, ques_limit, answer_limit, is_clue_topN) # func 0.1.1
#         examples_with_info.append(new_e)
#         # print(new_e)  #!!!
#         if debug and i >= debug_length:
#             break
#     if save_file is None:
#         save_file = file_type + "_answer_clue_style_info.pkl"
#     save(save_file, examples_with_info)
#     return examples_with_info


# def get_answer_clue_style_info(sentence, question, answer, answer_start,
#                                sent_limit=100, ques_limit=50, answer_limit=30, is_clue_topN=20): #GG func 0.1.1
#     chunklist, tree, doc = get_chunks(sentence) #func 0.1.1.1

#     # answer_info
#     answer_pos_tag = "UNK"
#     answer_ner_tag = "UNK"
#     for chunk in chunklist:
#         if answer == " ".join(chunk[2]):
#             answer_ner_tag = chunk[0]
#             answer_pos_tag = chunk[1]
#             break
#     answer_length = len(answer.split())

#     # question style info
#     question_type = get_question_type(question) # func 0.1.1.2

#     # clue info
#     clue_info = get_clue_info(
#         question, sentence, answer, answer_start,
#         chunklist=chunklist, y1_in_sent=None, doc=doc, ques_doc=None, sent_limit=sent_limit)# func 0.1.1.3

#     example = {
#         "question": question,
#         "ans_sent": sentence,
#         "answer_text": answer,
#         "question_type": question_type,
#         "answer_pos_tag": answer_pos_tag,
#         "answer_ner_tag": answer_ner_tag,
#         "answer_length": answer_length,
#         "clue_pos_tag": clue_info["clue_pos_tag"],
#         "clue_ner_tag": clue_info["clue_ner_tag"],
#         "clue_answer_dep_path_len": clue_info["clue_answer_dep_path_len"],
#         "clue_length": clue_info["clue_length"],
#         "clue_chunk": clue_info["clue_chunk"]
#     }

#     return example


# def get_token_char_level_spans(text, tokens):#GG use in get clue info

#     current = 0
#     spans = []
#     for token in tokens:
#         current = text.find(token, current)
#         if current < 0:
#             print("Token {} cannot be found".format(token))
#             raise Exception()
#         spans.append((current, current + len(token)))
#         current += len(token)
#     return spans


# In[16]:


# def get_clue_info(question, sentence, answer, answer_start,
#                   chunklist=None, y1_in_sent=None, doc=None, ques_doc=None, sent_limit=100): #GG func 0.1.1.3
#     example = {
#         "question": question,
#         "ans_sent": sentence,
#         "answer_text": answer}
#     if doc is None:
#         doc = NLP(sentence)
#     if ques_doc is None:
#         ques_doc = NLP(question)

#     if chunklist is None:
#         chunklist, _, _ = get_chunks(sentence, doc)

#     example["ans_sent_tokens"] = [token.text for token in doc]
#     example["ques_tokens"] = [token.text for token in ques_doc]
#     example["ans_sent_doc"] = doc

#     if y1_in_sent is None:
#         spans = get_token_char_level_spans(sentence, example["ans_sent_tokens"])
# #         [(0, 2), (3, 5), (6, 7), (8, 15), (16, 18), (19, 22), (23, 29), (30, 32), (33, 40), (40, 41), (42, 48), (49, 54), (55, 58), (59, 65), (66, 70), (71, 80), (81, 89), (90, 92), (93, 98), (99, 109), (110, 119), (120, 122), (123, 127), (127, 128)]
#         answer_end = answer_start + len(answer)
#         answer_span = []
#         for idx, span in enumerate(spans):
#             if not (answer_end <= span[0] or answer_start >= span[1]):
#                 answer_span.append(idx)

#         y1_in_sent = answer_span[0]

#     answer_start = y1_in_sent

#     doc_token_list = [token for token in doc]
#     idx2token, idx2related, context_tokens = get_all_related(doc, doc_token_list)

#     clue_rank_scores = []
#     for chunk in chunklist:
#         candidate_clue = chunk[2]  # list of chunk words

#         ques_lower = " ".join(example["ques_tokens"]).lower()
#         candidate_clue_text = " ".join(candidate_clue).lower()
#         ques_lemmas = [t.lemma_ for t in ques_doc]
#         sent_lemmas = [t.lemma_ for t in doc]
#         ques_tokens = [t.lower() for t in example["ques_tokens"]]
#         candidate_clue_is_content = [int(w.lower() not in FUNCTION_WORDS_LIST) for w in candidate_clue]
#         candidate_clue_lemmas = sent_lemmas[chunk[3]:chunk[4] + 1]
#         candidate_clue_content_lemmas = [candidate_clue_lemmas[i] for i in range(len(candidate_clue_lemmas)) if candidate_clue_is_content[i] == 1]
#         candidate_clue_lemmas_in_ques = [candidate_clue_lemmas[i] for i in range(len(candidate_clue_lemmas)) if candidate_clue_lemmas[i] in ques_lemmas]
#         candidate_clue_content_lemmas_in_ques = [candidate_clue_content_lemmas[i] for i in range(len(candidate_clue_content_lemmas)) if candidate_clue_content_lemmas[i] in ques_lemmas]

#         candidate_clue_tokens_in_ques = [candidate_clue[i] for i in range(len(candidate_clue)) if candidate_clue[i].lower() in ques_tokens]
#         candidate_clue_content_tokens = [candidate_clue[i] for i in range(len(candidate_clue)) if candidate_clue_is_content[i] == 1]
#         candidate_clue_content_tokens_in_ques = [candidate_clue_content_tokens[i] for i in range(len(candidate_clue_content_tokens)) if candidate_clue_content_tokens[i].lower() in ques_tokens]
#         candidate_clue_content_tokens_in_ques_soft = candidate_clue_content_tokens_in_ques  # !!!! TODO: soft in.

#         score = 0
#         if (len(candidate_clue_lemmas_in_ques) == len(candidate_clue_lemmas) or len(candidate_clue_tokens_in_ques) == len(candidate_clue)) and \
#                 sum(candidate_clue_is_content) > 0 and \
#                 candidate_clue[0].lower() not in NOT_BEGIN_TOKENS_FOR_ANSWER_CLUE:
#             score += len(candidate_clue_content_lemmas_in_ques)
#             score += len(candidate_clue_content_tokens_in_ques)
#             score += len(candidate_clue_content_tokens_in_ques_soft)
#             score += int(candidate_clue_text in ques_lower)

#         clue_rank_scores.append(score)
#         # print("______".join([str(score), " ".join(chunk[2])]))  #!!!!!!!!! for debug

#     if len(clue_rank_scores) == 0 or max(clue_rank_scores) == 0:
#         clue_chunk = None
#         clue_pos_tag = "UNK"
#         clue_ner_tag = "UNK"
#         clue_length = 0

#         clue_answer_dep_path_len = -1

#         selected_clue_binary_ids_padded = np.zeros([sent_limit], dtype=np.float32)
#     else:
#         clue_chunk = chunklist[clue_rank_scores.index(max(clue_rank_scores))]
#         clue_pos_tag = clue_chunk[1]
#         clue_ner_tag = clue_chunk[0]
#         clue_length = clue_chunk[4] - clue_chunk[3] + 1

#         clue_start = clue_chunk[3]
#         clue_end = clue_chunk[4]
#         clue_answer_dep_path_len = abs(clue_start - answer_start)
#         answer_related = idx2related[answer_start]
#         for tk_id, path in answer_related:
#             if tk_id == clue_start:
#                 clue_answer_dep_path_len = len(path)

#         selected_clue_binary_ids_padded = np.zeros([sent_limit], dtype=np.float32)
#         if clue_start < sent_limit and clue_end < sent_limit:
#             selected_clue_binary_ids_padded[clue_start:clue_end + 1] = 1

#     clue_info = {
#         "clue_pos_tag": clue_pos_tag,
#         "clue_ner_tag": clue_ner_tag,
#         "clue_length": clue_length,
#         "clue_chunk": clue_chunk,
#         "clue_answer_dep_path_len": clue_answer_dep_path_len,
#         "selected_clue_binary_ids_padded": selected_clue_binary_ids_padded,
#     }
#     return clue_info

# filename = 'train.txt'
# save_dataset_info_file = "data_set_inf.pkl"
# save_sample_probs_file = "data_set_probs.pkl"
# sample_probs = get_sample_probs(
#     filename, save_dataset_info_file, save_sample_probs_file,
#     sent_limit=100, ques_limit=50, answer_limit=30, is_clue_topN=20,
#     debug=True, debug_length=20,
#     answer_length_bin_width=3, answer_length_min_val=0, answer_length_max_val=30,
#     clue_dep_dist_bin_width=2, clue_dep_dist_min_val=0, clue_dep_dist_max_val=20)
# print("... above tests get_sample_probs\n")



import random
QUESTION_TYPES = [
    "Who", "Where", "When", "Why", "Which", "What", "How",
    "Boolean", "Other"]

# def get_question_type(question):# GG func 0.1.1.2 có thể thay đổi = cái khác

#     words = question.split()
#     for word in words:
#         for i in range(len(INFO_QUESTION_TYPES)):
#             if INFO_QUESTION_TYPES[i].upper() in word.upper():
#                 return (INFO_QUESTION_TYPES[i],
#                         Q_TYPE2ID_DICT[INFO_QUESTION_TYPES[i]])
#     for i in range(len(BOOL_QUESTION_TYPES)):
#         if BOOL_QUESTION_TYPES[i].upper() == words[0].upper():
#             return ("Boolean", Q_TYPE2ID_DICT["Boolean"])
#     return ("Other", Q_TYPE2ID_DICT["Other"])


# def get_token2char(doc):#GG func1.2
#     token2idx = {}
#     idx2token = {}
#     for token in doc:
#         token2idx[token.i] = token.idx, token.idx + len(token.text) - 1
#         for i in range(token.idx, token.idx + len(token.text)):
#             idx2token[i] = token.i
#     return token2idx, idx2token


# def val2bin(input_val, min_val, max_val, bin_width): #GG func1.3, 3.2, 0.2, 0.3
#     if input_val <= max_val and input_val >= min_val:
#         return math.ceil((input_val - min_val) / bin_width)
#     elif input_val > max_val:
#         return math.ceil((max_val - min_val) / bin_width) + 1
#     else:
#         return -1

# def get_chunks(sentence, doc=None):#GG func1.2, 0.1.1.1

#     tree = None
#     try:
#         tree = PARSER.parse(sentence)  # NOTICE: when sentence is too long, it will have error.
#     except:
#         pass
#     if doc is None:
#         doc = NLP(sentence)
#     max_depth, node_num, orig_chunklist = _navigate(tree)

#     chunklist = []
#     # parse the result of _navigate
#     for chunk in orig_chunklist:
#         try:
#             if chunk[0] == 'word':
#                 continue
#             chunk_pos_tag, leaves = chunk
#             leaves_without_position = []
#             position_list = []
#             for v in leaves:
#                 tmp = v.split('___')
#                 wd = tmp[0]
#                 index = int(tmp[1])
#                 leaves_without_position.append(wd)
#                 position_list.append(index)
#             st = position_list[0]
#             ed = position_list[-1]

#             chunk_ner_tag = "UNK"
#             chunk_text = " ".join(leaves_without_position)
#             for ent in doc.ents:
#                 if ent.text == chunk_text or chunk_text in ent.text:
#                     chunk_ner_tag = ent.label_

#             chunklist.append((chunk_ner_tag, chunk_pos_tag, leaves_without_position, st, ed))
#         except:
#             continue

#     return chunklist, tree, doc


# def get_all_related(context_doc, doc_token_list): #GG func 3.1

#     idx2token = {}
#     idx2related = {}
#     tokens = []
#     for token in context_doc:
#         idx2token[token.i] = token
#         related = {}
#         _dfs(context_doc, doc_token_list, token.i, [], len(context_doc) - 1, related)
#         sort_related = sorted(related.items(), key=lambda x: len(x[1]))
#         idx2related[token.i] = sort_related
#         tokens.append(token.text)
#     return idx2token, idx2related, tokens

# def _navigate(node):#GG
#     if type(node) is not nltk.Tree:
#         return 1, 1, [('word', node)]
#     # add position info of this node
#     for idx, _ in enumerate(node.leaves()):
#         tree_location = node.leaf_treeposition(idx)
#         non_terminal = node[tree_location[:-1]]
#         non_terminal[0] = non_terminal[0] + "___" + str(idx)

#     max_depth = 0
#     word_num = 0
#     chunklist = []
#     for child in node:
#         child_depth, child_num, child_chunklist = _navigate(child)
#         max_depth = max(max_depth, child_depth + 1)
#         word_num += child_num
#         chunklist += child_chunklist
#     cur_node_chunk = [(node.label(), node.leaves())]
#     chunklist += cur_node_chunk
#     return max_depth, word_num, chunklist


# def _post(orig_chunklist):
#     chunklist = []


#     for chunk in orig_chunklist:
#         try:
#             if chunk[0] == 'word':# like type of word = chunk[0] can N, adv, adj
#                 continue
#             label, leaves = chunk
#             leaves_without_position = []
#             position_list = []
#             for v in leaves:
#                 tmp = v.split('___')     
#                 wd = tmp[0]
#                 index = int(tmp[1])
#                 leaves_without_position.append(wd)
#                 position_list.append(index)
#             st = position_list[0]
#             ed = position_list[-1]
#             chunklist.append((label, leaves_without_position, st, ed))
#         except:
#             continue
#     cand_chunks = []
#     vp_list = [] # động từ
#     for chunk in chunklist:
#         label, leaves, st, ed = chunk
#         if label == 'VP':
#             vp_list.append(chunk)
#         else:
#             if label == 'PP':
#                 if leaves[0] in ['of', 'OF', 'for', 'FOR', 'to', 'TO']:
#                     continue
#             cand_chunks.append(chunk)

#     # delete VP which is not minimum chunk
#     for chunk in vp_list:
#         label, leaves, st, ed = chunk
#         cur_str = ' '.join(leaves)
#         contain_other = False
#         for other_chunk in vp_list:
#             other_leaves = other_chunk[1]
#             if other_leaves == leaves:
#                 continue
#             other_str = ' '.join(other_leaves)
#             if cur_str.find(other_str) >= 0:
#                 contain_other = True
#                 break
#         if not contain_other:
#             cand_chunks.append(chunk)

#     # other strategy of deleting chunk
#     answer_chunks = []
#     for chunk in chunklist:
#         label, leaves, st, ed = chunk
#         if label not in ['NP', 'VP', 'PP', 'UCP', 'ADVP']:
#             continue
#             pass
#         answer_chunks.append(chunk)
#     return answer_chunks

# def str_find(text, tklist): #GG func1.5
#     str_tk = ''
#     for tk in tklist:
#         str_tk += tk
#     tk1 = tklist[0]
#     pos = text.find(tk1)
#     while pos < len(text) and pos >= 0:
#         i, j = pos, 0
#         while i < len(text) and j < len(str_tk):
#             if text[i] == ' ':
#                 i += 1
#                 continue
#             if text[i] == str_tk[j]:
#                 i += 1
#                 j += 1
#             else:
#                 break
#         if j == len(str_tk):
#             return pos, i - 1
#         newpos = text[pos + 1:].find(tk1)
#         if newpos >= 0:
#             pos = pos + 1 + newpos
#         else:
#             break
#     return -1, -1


# def select_answers(context, processed_by_spacy=False):

#     tree = None
#     try:
#         tree = PARSER.parse(context)  # TODO: if the context is too long, it will cause error.
#     except:
#         pass
#     if not processed_by_spacy:
#         doc = NLP(context)
#     else:
#         doc = context
#     token2idx, idx2token = get_token2char(doc)
#     max_depth, node_num, chunklist = _navigate(tree)
#     answer_chunks = _post(chunklist)
#     answers = []
#     for chunk in answer_chunks:
#         label, leaves, st, ed = chunk
#         # print('leaves={}\tst={}\ted={}\tlabel={}'.format(leaves, st, ed, label))
#         try:
#             char_st, char_ed = str_find(context, leaves)
#             if char_st < 0:
#                 continue
#             answer_text = context[char_st:char_ed + 1]
#             st = idx2token[char_st]
#             ed = idx2token[char_ed]
#             answer_bio_ids = ['O'] * len(doc)
#             answer_bio_ids[st: ed + 1] = ['I'] * (ed - st + 1)
#             answer_bio_ids[st] = 'B'
#             char_st = token2idx[st][0]
#             char_ed = token2idx[ed][1]
#             # print('answer_text={}\tchar_st={}\tchar_ed={}\tst={}\ted={}'.format(answer_text, char_st, char_ed, st, ed))
#         except:
#             continue
#         answers.append((answer_text, char_st, char_ed, answer_bio_ids, label))

#     return answers


# def weighted_sample(choices, probs):#GG func 1.4, 2.1, 3.3

#     probs = [x / sum(probs) for x in probs]
#     probs = np.concatenate(([0], np.cumsum(probs)))
#     r = random.random()
#     for j in range(len(choices) + 1):
#         if probs[j] < r <= probs[j + 1]:
#             return choices[j]

# def augment_qg_data(sentence, sample_probs,
#                     num_sample_answer=5, num_sample_clue=2, num_sample_style=2,
#                     answer_length_bin_width=3, answer_length_min_val=0, answer_length_max_val=30,
#                     clue_dep_dist_bin_width=2, clue_dep_dist_min_val=0, clue_dep_dist_max_val=20,
#                     max_sample_times=20):#GG main

#     sampled_infos = []

#     # sample answer chunk
#     sampled_answers, chunklist, tree, doc = select_answers(
#         sentence, sample_probs,
#         num_sample_answer,
#         answer_length_bin_width, answer_length_min_val, answer_length_max_val,
#         max_sample_times)# func 1

#     for ans in sampled_answers:
#         (answer_text, char_st, char_ed, st, ed, answer_bio_ids, answer_pos_tag, answer_ner_tag) = ans
#         info = {"answer": {"answer_text": answer_text, "char_start": char_st, "char_end": char_ed,
#                            "answer_bio_ids": answer_bio_ids, "answer_chunk_tag": answer_pos_tag},
#                 "styles": None,
#                 "clues": None}
#         # sample style
#         styles = select_question_types(sample_probs, ans,
#                                        num_sample_style,
#                                        max_sample_times) #func 2
#         info["styles"] = list(styles)

#         # sample clue
#         clues = select_clues(chunklist, doc, sample_probs, ans,
#                              num_sample_clue,
#                              clue_dep_dist_bin_width, clue_dep_dist_min_val, clue_dep_dist_max_val,
#                              max_sample_times) # func 3
#         info["clues"] = clues

#         sampled_infos.append(info)

#     result = {"context": sentence, "selected_infos": sampled_infos, "ans_sent_doc": doc}
#     return result


# def select_answers(sentence, sample_probs,
#                    num_sample_answer=5,
#                    answer_length_bin_width=3, answer_length_min_val=0, answer_length_max_val=30,
#                    max_sample_times=20):#GG func 1
#     # get all chunks
#     chunklist, tree, doc = get_chunks(sentence) #func1.1
#     token2idx, idx2token = get_token2char(doc) #func1.2

#     # sample answer chunk
#     chunk_ids = list(range(len(chunklist)))
#     a_probs = []
#     for chunk in chunklist:
#         chunk_pos_tag = chunk[1]
#         chunk_ner_tag = chunk[0]
#         a_tag = "-".join([chunk_pos_tag, chunk_ner_tag])
#         a_length = abs(chunk[3] - chunk[4] + 1)
#         a_length_bin = val2bin(a_length, answer_length_min_val, answer_length_max_val, answer_length_bin_width) # func1.3
#         a_condition = "_".join([a_tag, str(a_length_bin)])  # condition of p(a|...)
#         if a_condition in sample_probs["a"] and chunk[2][0].lower() not in NOT_BEGIN_TOKENS_FOR_ANSWER_CLUE:
#             a_probs.append(sample_probs["a"][a_condition])
#         else:
#             a_probs.append(1)

#     sampled_answer_chunk_ids = []
#     sample_times = 0
#     for sample_times in range(max_sample_times):
#         sampled_chunk_id = weighted_sample(chunk_ids, a_probs) # func1.4
#         if sampled_chunk_id not in sampled_answer_chunk_ids:
#             sampled_answer_chunk_ids.append(sampled_chunk_id)
#         if len(sampled_answer_chunk_ids) >= num_sample_answer:
#             break

#     sampled_answers = []
#     for chunk_id in sampled_answer_chunk_ids:
#         chunk = chunklist[chunk_id]
#         chunk_ner_tag, chunk_pos_tag, leaves, st, ed = chunk
#         try:
#             context = sentence
#             char_st, char_ed = str_find(context, leaves)#func1.5
#             if char_st < 0:
#                 continue
#             answer_text = context[char_st:char_ed + 1]
#             st = idx2token[char_st]
#             ed = idx2token[char_ed]
#             answer_bio_ids = ['O'] * len(doc)
#             answer_bio_ids[st: ed + 1] = ['I'] * (ed - st + 1)
#             answer_bio_ids[st] = 'B'
#             char_st = token2idx[st][0]
#             char_ed = token2idx[ed][1]
#             sampled_answers.append((answer_text, char_st, char_ed, st, ed, answer_bio_ids, chunk_pos_tag, chunk_ner_tag))
#         except:
#             continue

#     return sampled_answers, chunklist, tree, doc


# def select_question_types(sample_probs, selected_answer,
#                           num_sample_style=2,
#                           max_sample_times=20): #GG func2
#     (answer_text, char_st, char_ed, st, ed, answer_bio_ids, answer_pos_tag, answer_ner_tag) = selected_answer
#     a_tag = "-".join([answer_pos_tag, answer_ner_tag])

#     # get s probs
#     styles = QUESTION_TYPES  # question types
#     s_probs = []
#     for s in QUESTION_TYPES:
#         s_condition = "_".join([s, a_tag])
#         if s_condition in sample_probs["s|c,a"]:
#             s_probs.append(sample_probs["s|c,a"][s_condition])
#         else:
#             s_probs.append(1)

#     # sample s
#     sampled_styles = []
#     sample_times = 0
#     for sample_times in range(max_sample_times):
#         sampled_s = weighted_sample(styles, s_probs) #func 2.1
#         if sampled_s not in sampled_styles:
#             sampled_styles.append(sampled_s)
#         if len(sampled_styles) >= num_sample_style:
#             break

#     return sampled_styles

# def select_clues(chunklist, doc, sample_probs, selected_answer,
#                  num_sample_clue=2,
#                  clue_dep_dist_bin_width=2, clue_dep_dist_min_val=0, clue_dep_dist_max_val=20,
#                  max_sample_times=20): # GG func 3
#     (answer_text, char_st, char_ed, st, ed, answer_bio_ids, answer_pos_tag, answer_ner_tag) = selected_answer

#     doc_token_list = [token for token in doc]  # doc is sentence_doc
#     idx2token, idx2related, context_tokens = get_all_related(doc, doc_token_list) # func 3.1

#     # get c probs
#     c_probs = []
#     for chunk in chunklist:
#         chunk_pos_tag = chunk[1]
#         chunk_ner_tag = chunk[0]
#         c_tag = "-".join([chunk_pos_tag, chunk_ner_tag])

#         answer_start = st
#         clue_start = chunk[3]
#         clue_end = chunk[4]
#         clue_answer_dep_path_len = abs(clue_start - answer_start)
#         answer_related = idx2related[answer_start]
#         for tk_id, path in answer_related:
#             if tk_id == clue_start:
#                 clue_answer_dep_path_len = len(path)
#         dep_dist = clue_answer_dep_path_len

#         dep_dist_bin = val2bin(dep_dist, clue_dep_dist_min_val, clue_dep_dist_max_val, clue_dep_dist_bin_width) #func 3.2

#         c_condition = "_".join([c_tag, str(dep_dist_bin)])  # condition of p(c|...)
#         if c_condition in sample_probs["c|a"] and chunk[2][0].lower() not in NOT_BEGIN_TOKENS_FOR_ANSWER_CLUE:
#             c_probs.append(sample_probs["c|a"][c_condition])
#         else:
#             c_probs.append(1)

#     # sample c
#     chunk_ids = list(range(len(chunklist)))
#     sampled_clue_chunk_ids = []
#     sample_times = 0
#     for sample_times in range(max_sample_times):
#         sampled_chunk_id = weighted_sample(chunk_ids, c_probs)# func 3.3
#         if sampled_chunk_id not in sampled_clue_chunk_ids:
#             sampled_clue_chunk_ids.append(sampled_chunk_id)
#         if len(sampled_clue_chunk_ids) >= num_sample_clue:
#             break

#     sampled_clues = []
#     for chunk_id in sampled_clue_chunk_ids:
#         chunk = chunklist[chunk_id]
#         clue_start = chunk[3]
#         clue_end = chunk[4]
#         clue_text = ' '.join(context_tokens[clue_start:clue_end + 1])
#         clue_binary_ids = [0] * len(doc_token_list)
#         clue_binary_ids[clue_start:clue_end + 1] = [1] * (clue_end - clue_start + 1)
#         clue = {"clue_text": clue_text, "clue_binary_ids": clue_binary_ids}
#         sampled_clues.append(clue)

#     return sampled_clues
