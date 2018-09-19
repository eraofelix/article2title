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


class MainHandler(RequestHandler):
    def post(self):
        try:
            audio = self.get_argument('audio', '')
            print('------audio:', type(audio), np.shape(audio), audio)
        except:
            pass

    def get(self):
        article = RequestHandler.get_argument(self, name='article')


        article = translator.translate(article, src='auto', dest='en').text.lower().replace('.', ' .').replace(',', ' ,')

        print('---article:', article)

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

