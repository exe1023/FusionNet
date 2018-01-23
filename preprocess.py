import json
import spacy
import os
from tqdm import tqdm
import jieba
import tokenizers
from functools import partial
from multiprocessing import Pool
from multiprocessing.util import Finalize

def tokenize_ch(text):
    text = jieba.cut(text.strip(), cut_all=False)
    return list(text)

def preprocess_ch(file_name, output_name, skip_answer=False):
    '''
    preprocess ch data
    '''
    data = json.load(open(file_name, 'r'))
    examples = []
    for article in tqdm(data['data']):
        # keys: title, paragraphs
        for paragraph in article['paragraphs']:
            # keys: context, qas
            context = paragraph['context']
            context_token = tokenize_ch(context)
            for qa in paragraph['qas']:
                qa_id = qa['id']
                q = qa['question']
                q_token = tokenize_ch(q)
                if skip_answer:
                    ans_text, ans_text_token, ans_start, ans_end = '', [], -1, -1
                else:
                    ans_text = qa['answers'][0]['text']
                    ans_text_token = tokenize_ch(ans_text)
                    ans_start = qa['answers'][0]['answer_start']
                    ans_end = ans_start + len(ans_text)
                assert context[ans_start:ans_end] == ans_text
                examples.append({
                                'id': qa_id,
                                'context': context,
                                'context_token': context_token,
                                'q': q,
                                'q_token': q_token,
                                'ans_text': ans_text,
                                'ans_text_token': ans_text_token,
                                'ans_start': ans_start,
                                'ans_end': ans_end
                })
    json.dump(examples, open(output_name, 'w'))

TOK = None

def init(tokenizer_class, options):
    global TOK
    TOK = tokenizer_class(**options)
    Finalize(TOK, TOK.shutdown, exitpriority=100)

def load_squad(file_name):
    data = json.load(open(file_name, 'r'))['data']
    output = {'qids': [], 'questions': [], 'answers': [],
              'contexts': [], 'qid2cid': []}
    for article in data:
        for paragraph in article['paragraphs']:
            output['contexts'].append(paragraph['context'])
            for qa in paragraph['qas']:
                output['qids'].append(qa['id'])
                output['questions'].append(qa['question'])
                output['qid2cid'].append(len(output['contexts']) - 1)
                if 'answers' in qa:
                    output['answers'].append(qa['answers'])
    return output

def find_answer(offsets, begin_offset, end_offset):
    """Match token offsets with the char begin/end offsets of the answer."""
    start = [i for i, tok in enumerate(offsets) if tok[0] == begin_offset]
    end = [i for i, tok in enumerate(offsets) if tok[1] == end_offset]
    assert(len(start) <= 1)
    assert(len(end) <= 1)
    if len(start) == 1 and len(end) == 1:
        return start[0], end[0]

def tokenize(text):
    """Call the global process tokenizer on the input text."""
    global TOK
    tokens = TOK.tokenize(text)
    output = {
        'words': tokens.words(),
        'offsets': tokens.offsets(),
        'pos': tokens.pos(),
        'lemma': tokens.lemmas(),
        'ner': tokens.entities(),
    }
    return output

def preprocess_squad(file_name, output_name, skip_answer=False):
    data = load_squad(file_name)
    
    tokenizer_class = tokenizers.get_class('spacy')
    make_pool = partial(Pool, 8, initializer=init)
    workers = make_pool(initargs=(tokenizer_class, {'annotators': {'lemma'}}))
    q_tokens = workers.map(tokenize, data['questions'])
    workers.close()
    workers.join()

    workers = make_pool(
        initargs=(tokenizer_class, {'annotators': {'lemma', 'pos', 'ner'}})
    )
    c_tokens = workers.map(tokenize, data['contexts'])
    workers.close()
    workers.join()

    examples = []
    for idx in range(len(data['qids'])):
        question = q_tokens[idx]['words']
        qlemma = q_tokens[idx]['lemma']
        document = c_tokens[data['qid2cid'][idx]]['words']
        offsets = c_tokens[data['qid2cid'][idx]]['offsets']
        lemma = c_tokens[data['qid2cid'][idx]]['lemma']
        pos = c_tokens[data['qid2cid'][idx]]['pos']
        ner = c_tokens[data['qid2cid'][idx]]['ner']
        ans_tokens = []
        if len(data['answers']) > 0:
            for ans in data['answers'][idx]:
                found = find_answer(offsets,
                                    ans['answer_start'],
                                    ans['answer_start'] + len(ans['text']))
                if found:
                    ans_tokens.append(found)
        examples.append( {
            'id': data['qids'][idx],
            'question': question,
            'document': document,
            'offsets': offsets,
            'answers': ans_tokens,
            'qlemma': qlemma,
            'lemma': lemma,
            'pos': pos,
            'ner': ner,
        })
    json.dump(examples, open(output_name, 'w'))

if __name__ == '__main__':
    #preprocess_ch('CQA_data/train-v1.1.json', 'train.json')
    #preprocess_ch('CQA_data/test-v1.1.json', 'test.json', True)
    preprocess_squad('squad/train-v1.1.json', 'train_squad.json')
    preprocess_squad('squad/dev-v1.1.json', 'dev_squad.json')