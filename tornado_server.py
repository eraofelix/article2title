#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tornado.web import RequestHandler, Application
from tornado.ioloop import IOLoop


class MainHandler(RequestHandler):
    def get(self):
        article = RequestHandler.get_argument(self, name='article')
        print('artlcle reveieved:', article, ', title_pred is:---------')
        # test sum to get title
        self.write('artlcle reveieved, title_pred is:---------\n')


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