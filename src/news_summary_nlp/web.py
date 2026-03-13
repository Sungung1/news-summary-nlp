from pathlib import Path

import keras, torch, urllib.request, requests, re, os, json, random, numpy as np, pandas as pd
from flask import Flask, request, render_template, jsonify
from summarizer import Summarizer, TransformerSummarizer
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration, pipeline
from keras.callbacks import EarlyStopping, ModelCheckpoint
from konlpy.tag import Okt
from bs4 import BeautifulSoup
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate
from tensorflow.keras.models import Model
from io import StringIO
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = PROJECT_ROOT / "myproject"

encoder_model = keras.models.load_model(str(MODEL_ROOT / "enc_mode2"))
decoder_model = keras.models.load_model(str(MODEL_ROOT / "dec_model2"))
text_max_len = 350
summary_max_len = 30
stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다', '을', '이다', '다']

def news_preprocessing(sentence):
    okt = Okt()
    tokenized_sentence = okt.morphs(re.sub(r'[^0-9가-힣\s]', '', sentence), stem=True)  # 토큰화
    stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords]  # 불용어 제거
    result = ' '.join(stopwords_removed_sentence)
    return result

app = Flask(__name__, template_folder=str(MODEL_ROOT / "templates"))
app.debug = True

# Main page
@app.route('/')
def start():
    return render_template('main.html')

@app.route('/generate_summary', methods=['POST'])
def generate_summary():
    if request.method == 'POST':
        url = request.form.get('article_url')

        try:
            # URL에서 뉴스 본문 크롤링
            web = requests.get(url).content
            source = BeautifulSoup(web, 'html.parser')
            # 본문 요소를 찾아봅니다.
            article_body = source.find('div', {'id': 'newsct_article'})  # 또는 원하는 요소 찾아서 사용
            if article_body:
                news_content = article_body.get_text()
            else:
                return "뉴스 본문을 찾을 수 없습니다."
        except Exception as e:
            # 크롤링 과정에서 오류 발생
            return f"<h2>오류 발생: {str(e)}</h2>"

        # Bert 요약
        bert_model = Summarizer()
        summary_bert = bert_model(news_content)

        # Kobart 요약
        tokenizer = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')
        model = BartForConditionalGeneration.from_pretrained('digit82/kobart-summarization')

        news_content_kobart = news_content
        raw_input_ids = tokenizer.encode(news_content_kobart)
        input_ids = [tokenizer.bos_token_id] + raw_input_ids + [tokenizer.eos_token_id]
        summary_ids = model.generate(torch.tensor([input_ids]), num_beams=4, max_length=512, eos_token_id=1)
        summary_kobart = tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)

        news_content_our = news_preprocessing(news_content.replace("\n", ""))
        data = pd.read_csv(StringIO(news_content_our), header=None, names=["text"], lineterminator='\n')
        data['decoder_input'] = data['text'].apply(lambda x: 'sostoken ' + x)
        data['decoder_target'] = data['text'].apply(lambda x: x + ' eostoken')
        encoder_input = np.array(data['text'])
        decoder_input = np.array(data['decoder_input'])
        decoder_target = np.array(data['decoder_target'])
        tar_tokenizer = Tokenizer()
        tar_tokenizer.fit_on_texts(decoder_input)
        src_tokenizer = Tokenizer()
        src_tokenizer.fit_on_texts(encoder_input)
        tar_vocab = 7253
        tar_tokenizer = Tokenizer(num_words=tar_vocab)
        tar_tokenizer.fit_on_texts(decoder_input)
        tar_tokenizer.fit_on_texts(decoder_target)
        src_index_to_word = src_tokenizer.index_word  # 원문 단어 집합에서 정수 -> 단어를 얻음
        tar_word_to_index = tar_tokenizer.word_index  # 요약 단어 집합에서 단어 -> 정수를 얻음
        tar_index_to_word = tar_tokenizer.index_word  # 요약 단어 집합에서 정수 -> 단어를 얻음
        encoder_input_train = encoder_input[:]
        decoder_input_train = decoder_input[:]
        decoder_target_train = decoder_target[:]
        encoder_input_test = encoder_input[:]
        decoder_input_test = decoder_input[:]
        decoder_target_test = decoder_target[:]
        decoder_input_train = tar_tokenizer.texts_to_sequences(decoder_input_train)
        decoder_target_train = tar_tokenizer.texts_to_sequences(decoder_target_train)
        decoder_input_test = tar_tokenizer.texts_to_sequences(decoder_input_test)
        decoder_target_test = tar_tokenizer.texts_to_sequences(decoder_target_test)
        src_vocab = 28251
        src_tokenizer = Tokenizer(num_words=src_vocab)
        src_tokenizer.fit_on_texts(encoder_input_train)

        # 텍스트 시퀀스를 정수 시퀀스로 변환
        encoder_input_train = src_tokenizer.texts_to_sequences(encoder_input_train)
        encoder_input_test = src_tokenizer.texts_to_sequences(encoder_input_test)
        drop_train = [index for index, sentence in enumerate(decoder_input_train) if len(sentence) == 1]
        drop_test = [index for index, sentence in enumerate(decoder_input_test) if len(sentence) == 1]
        encoder_input_train = np.delete(encoder_input_train, drop_train, axis=0)
        decoder_input_train = np.delete(decoder_input_train, drop_train, axis=0)
        decoder_target_train = np.delete(decoder_target_train, drop_train, axis=0)

        encoder_input_test = np.delete(encoder_input_test, drop_test, axis=0)
        decoder_input_test = np.delete(decoder_input_test, drop_test, axis=0)
        decoder_target_test = np.delete(decoder_target_test, drop_test, axis=0)
        encoder_input_train = pad_sequences(encoder_input_train, maxlen=text_max_len, padding='post')
        encoder_input_test = pad_sequences(encoder_input_test, maxlen=text_max_len, padding='post')
        decoder_input_train = pad_sequences(decoder_input_train, maxlen=summary_max_len, padding='post')
        decoder_target_train = pad_sequences(decoder_target_train, maxlen=summary_max_len, padding='post')
        decoder_input_test = pad_sequences(decoder_input_test, maxlen=summary_max_len, padding='post')
        decoder_target_test = pad_sequences(decoder_target_test, maxlen=summary_max_len, padding='post')

        def seq2text(input_seq):
            sentence = ''
            for i in input_seq:
                if (i != 0):
                    sentence = sentence + src_index_to_word[i] + ' '
            return sentence

        # 요약문의 정수 시퀀스를 텍스트 시퀀스로 변환
        def seq2summary(input_seq):
            sentence = ''
            for i in input_seq:
                if ((i != 0 and i != tar_word_to_index['sostoken']) and i != tar_word_to_index['eostoken']):
                    sentence = sentence + tar_index_to_word[i] + ' '
            return sentence

        def decode_sequence(input_seq):
            # 입력으로부터 인코더의 상태를 얻음
            e_out, e_h, e_c = encoder_model.predict(input_seq)

            # <SOS>에 해당하는 토큰 생성
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = tar_word_to_index['sostoken']

            stop_condition = False
            decoded_sentence = ''
            while not stop_condition:  # stop_condition이 True가 될 때까지 루프 반복

                output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])
                sampled_token_index = np.argmax(output_tokens[0, -1, :])
                sampled_token = tar_index_to_word[sampled_token_index]

                if (sampled_token != 'eostoken'):
                    decoded_sentence += ' ' + sampled_token

                #  <eos>에 도달하거나 최대 길이를 넘으면 중단.
                if (sampled_token == 'eostoken' or len(decoded_sentence.split()) >= (summary_max_len - 1)):
                    stop_condition = True

                # 길이가 1인 타겟 시퀀스를 업데이트
                target_seq = np.zeros((1, 1))
                target_seq[0, 0] = sampled_token_index

                # 상태를 업데이트 합니다.
                e_h, e_c = h, c

            return decoded_sentence

        our = seq2summary(decoder_input_test[0])  # 우리꺼 요약문
        # HTML 템플릿 렌더링 및 요약 내용 전달
        return render_template('main.html', news_content=news_content, summary_bert=summary_bert, summary_kobart=summary_kobart, our=our)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
