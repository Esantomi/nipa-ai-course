# elice-utils
# maintainer: Suin Kim (suin@elicer.com) and Jungkook Park (jk@elicer.com)

import base64
import mimetypes
import os
import urllib.parse
import urllib.request


class EliceUtils(object):

    def __init__(self):
        self._execution_token = os.getenv('EXECUTION_TOKEN')
        self._executor_ip = os.getenv('EXECUTOR_IP')
        self._executor_com_port = os.getenv('EXECUTOR_COM_PORT')
        self._otp_key = None

        self._local_mode = False

        if not all((self._execution_token, self._executor_ip, self._executor_com_port)):
            self._local_mode = True

            print('=== NON-ELICE ENVIRONMENT ===')
            print('Warning: This script is running on the non-elice environment. '
                  'All outputs will be redirected to standard output.')
            print('=============================')

    def _send(self, url, data):
        if self._local_mode:
            msg_type = data['type']
            msg_data = data['data']

            if msg_type in ['grader', 'score']:
                print('[%s] %s' % (msg_type, msg_data), end='')
            else:
                print('[%s]' % msg_type, end='')

            return

        data_encoded = urllib.parse.urlencode(data)
        q = urllib.request.Request(url,
                                   data=data_encoded.encode('utf-8'))

        try:
            urllib.request.urlopen(q)
        except Exception:
            raise Exception('Failed to send message to elice.')

    def _handle_image(self, filepath):
        mtype, _ = mimetypes.guess_type(filepath)

        if mtype is None or not mtype.startswith('image/'):
            raise ValueError('Invalid image filepath.')

        with open(filepath, 'rb') as f:
            data = 'data:%s;base64,%s' % (
                mtype,
                base64.b64encode(f.read()).decode('utf-8')
            )

        return data

    def _handle_file(self, filepath):
        mtype, _ = mimetypes.guess_type(filepath)

        with open(filepath, 'rb') as f:
            data = '%s;data:%s;base64,%s' % (
                os.path.basename(filepath),
                mtype or 'application/octet-stream',
                base64.b64encode(f.read()).decode('utf-8')
            )

        return data

    def send(self, msg_type, msg_data):
        self._send(
            'http://%s:%s/comm/send/%s' % (self._executor_ip,
                                           self._executor_com_port,
                                           self._execution_token),
            {'type': msg_type, 'data': msg_data}
        )

    def send_image(self, filepath):
        self.send('image', self._handle_image(filepath))

    def send_file(self, filepath):
        self.send('file', self._handle_file(filepath))

    def secure_init(self):
        if self._local_mode:
            return

        try:
            r = urllib.request.urlopen(
                'http://%s:%s/comm/secure/init/%s' % (self._executor_ip,
                                                      self._executor_com_port,
                                                      self._execution_token)
            )
        except Exception:
            raise Exception('Failed to initialize elice util secure channel.')

        self._otp_key = r.read().decode('utf-8')

    def secure_send(self, msg_type, msg_data):
        self._send(
            'http://%s:%s/comm/secure/send/%s/%s' % (self._executor_ip,
                                                     self._executor_com_port,
                                                     self._execution_token,
                                                     self._otp_key),
            {'type': msg_type, 'data': msg_data}
        )

    def secure_send_image(self, filepath):
        self.secure_send('image', self._handle_image(filepath))

    def secure_send_file(self, filepath):
        self.secure_send('file', self._handle_file(filepath))

    def secure_send_grader(self, msg):
        self.secure_send('grader', msg)

    def secure_send_score(self, score):
        self.secure_send('score', score)
