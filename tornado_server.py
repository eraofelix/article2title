#!/usr/bin/env python
# -*- coding: utf-8 -*-
from threading import Thread, Lock
from tornado.web import RequestHandler, Application
from tornado.ioloop import IOLoop
import tensorflow as tf
import pickle
from model import Model
from utils import build_dict, build_dataset, batch_iter, build_deploy
import time
from googletrans import Translator
import numpy as np
import IdentificationServiceHttpClientHelper
import os
import sounddevice as sd
import soundfile as sf

translator = Translator()
lock = Lock()
# 1, build model
with open("args.pickle", "rb") as f:
    args = pickle.load(f)

print("Loading dictionary...")
word_dict, reversed_dict, article_max_len, summary_max_len = build_dict("valid", args.toy)
sess = tf.InteractiveSession()
print("Loading saved model...")
t1 = time.time()
model = Model(reversed_dict, article_max_len, summary_max_len, args, forward_only=True)
saver = tf.train.Saver(tf.global_variables())
ckpt = tf.train.get_checkpoint_state("./saved_model/")
saver.restore(sess, ckpt.model_checkpoint_path)
print('load model time:', str(time.time() - t1) + 's')

filePath = 'tmp.wav'
fs = 16000

profile_ids = ['aabd9804-3c66-46d1-b8d3-4598b8aca4d8',
                            '7874fb49-1c07-493c-b948-a496d6b1d1a9',
                            'cce7883d-51b8-45aa-8474-1e6b987e5dbf',
                            'cb86bd34-55fb-41b2-8995-02c6f170672a',
                            'cc4c4bee-adab-47f1-91f3-d45b961f09c5']
profile_nms = ['kai', 'kun', 'wenpeng', 'gongjing', 'zhengwei']

class MainHandler(RequestHandler):
    def post(self):
        print('enter post...')
        try:
            audio = RequestHandler.get_body_argument(self, name='audio')  # '1,1,1,...,11,2'
            # print('------audio:', type(audio), np.shape(audio), str(audio))
            # self.write(str(audio) + '\n')

            audio_lst_str = audio.split(',')  # [str*32000]
            audio_lst_int = [float(i) for i in audio_lst_str]
            print('------audio_lst_int:', type(audio_lst_int), np.shape(audio_lst_int))
            audio_array = np.asarray(audio_lst_int)
            audio_array = np.expand_dims(audio_array, axis=1)
            print('------audio_array:', type(audio_array), np.shape(audio_array))

            try:
                os.remove(filePath)
            except OSError:
                pass
            print('1')
            sf.write(filePath, audio_array, fs)
            print('2')
            helper = IdentificationServiceHttpClientHelper.IdentificationServiceHttpClientHelper('ccc2411ed1bb496fbc3aaf42540e81ac')
            print('3')
            identification_response = helper.identify_file(filePath, profile_ids, 'true')
            print('4')
            id = identification_response.get_identified_profile_id()
            print('5')
            print('current profile_id:', id)

            if id in profile_ids:
                print('id:', id)
                idx = profile_ids.index(id)
                print('idx:', idx)
                name = profile_nms[idx]
                print('声纹鉴定结果:', name)
                print('鉴定confidence：', identification_response.get_confidence())
                self.write(name + '\n')
            else:
                print('id not in profile_ids')

            # name = profile_nms[idx] if id in profile_ids else 'stranger'
            # print('声纹鉴定结果:', name)
            # print('鉴定confidence：', identification_response.get_confidence())
            # self.write(name + '\n')

        except:
            print('receieve post but something error')
            pass

    def get(self):
        article = RequestHandler.get_argument(self, name='article')
        print('enter get...')
        try:
            print('article in get:', article)
        except:
            print('receieve get but cannot print')
            pass
        article = translator.translate(article, src='auto', dest='en').text.lower().replace('.', ' .').replace(',', ' ,')
        # print('---article:', article)

        print("Loading dictionary...")
        word_dict, reversed_dict, article_max_len, summary_max_len = build_dict("valid", args.toy)
        valid_x, valid_y = build_deploy(article, word_dict, article_max_len, summary_max_len)
        valid_x_len = list(map(lambda x: len([y for y in x if y != 0]), valid_x))

        batches = batch_iter(valid_x, valid_y, args.batch_size, 1)
        print("Start auto summarization...")
        for batch_x, batch_y in batches:
            batch_x_len = list(map(lambda x: len([y for y in x if y != 0]), batch_x))
            valid_feed_dict = {
                model.batch_size: len(batch_x),
                model.X: batch_x,
                model.X_len: batch_x_len,
            }
            t0 = time.time()
            prediction = sess.run(model.prediction, feed_dict=valid_feed_dict)
            prediction_output = list(map(lambda x: [reversed_dict[y] for y in x], prediction[:, 0, :]))

            print('inference time:', str(time.time() - t0) + 's')

            line = prediction_output[0]
            summary = list()
            for word in line:
                if word == "</s>":
                    break
                if word not in summary:
                    summary.append(word)
            title_pred = " ".join(summary)
            print('title_pred:', title_pred)
            title_cn = translator.translate(title_pred, src='auto', dest='zh-cn').text
            # print('title_cn:', title_cn)
            self.write(str(title_cn) + '\n')


if __name__ == '__main__':
    app = Application(
        [
            (r'/', MainHandler)
        ]
    )

    port = 8008
    app.listen(port=port, address='0.0.0.0')
    print('listening to port:', port)

    IOLoop.current().start()

