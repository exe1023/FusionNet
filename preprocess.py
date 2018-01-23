import json
import spacy
import os
from tqdm import tqdm
import jieba
import tokenizers
dict_path = os.path.join(os.getenv("JIEBA_DATA"), "dict.txt.big") 

annotators = {'ner','pos'}
nlp = tokenizers.get_class('corenlp')(annotators=annotators)

def tokenize(text):
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
            context_token = tokenize(context)
            for qa in paragraph['qas']:
                qa_id = qa['id']
                q = qa['question']
                q_token = tokenize(q)
                if skip_answer:
                    ans_text, ans_text_token, ans_start, ans_end = '', [], -1, -1
                else:
                    ans_text = qa['answers'][0]['text']
                    ans_text_token = tokenize(ans_text)
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

def preprocess_squad(file_name, output_name, skip_answer=False):
    data = load_squad(file_name)
    import spacy
    nlp = spacy.load('en')

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
        yield {
            'id': data['qids'][idx],
            'question': question,
            'document': document,
            'offsets': offsets,
            'answers': ans_tokens,
            'qlemma': qlemma,
            'lemma': lemma,
            'pos': pos,
            'ner': ner,
        }

if __name__ == '__main__':
    preprocess_ch('CQA_data/train-v1.1.json', 'train.json')
    preprocess_ch('CQA_data/test-v1.1.json', 'test.json', True)