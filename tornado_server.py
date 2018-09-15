#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tornado.web import RequestHandler, Application
from tornado.ioloop import IOLoop


def get_title(article):

    return 'not implement yet~'


class MainHandler(RequestHandler):
    def get(self):
        article = RequestHandler.get_argument(self, name='article')
        print('article reveieved:' + str(article))
        # test sum to get title
        title = get_title(article)
        self.write(str(article) + '->'+ title+'\n')


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