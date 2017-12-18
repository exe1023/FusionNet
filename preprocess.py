import json
import spacy
import os
from tqdm import tqdm
import jieba
dict_path = os.path.join(os.getenv("JIEBA_DATA"), "dict.txt.big") 


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

if __name__ == '__main__':
    preprocess_ch('CQA_data/train-v1.1.json', 'train.json')
    preprocess_ch('CQA_data/test-v1.1.json', 'test.json', True)