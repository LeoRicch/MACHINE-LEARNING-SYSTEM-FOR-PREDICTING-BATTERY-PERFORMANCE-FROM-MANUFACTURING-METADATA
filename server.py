"""Loopback-only web interface for the local Casimir engine."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
import threading
from urllib.parse import urlparse
import webbrowser

from casimir.service import Engine

ROOT = Path(__file__).resolve().parent


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, engine, **kwargs):
        self.engine = engine
        super().__init__(*args, directory=str(ROOT/'web'), **kwargs)

    def _json(self, payload, status=200):
        content = json.dumps(payload, allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(content)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(content)

    def _local_request(self):
        # Reject foreign host/origin requests. The app never exposes a LAN listener.
        port = self.server.server_port
        expected = {f'127.0.0.1:{port}', f'localhost:{port}'}
        if self.headers.get('Host') not in expected:
            return False
        origin = self.headers.get('Origin')
        return not origin or origin in {f'http://{host}' for host in expected}

    def do_GET(self):
        if not self._local_request():
            return self._json({'error':'Local app requests only.'}, 403)
        path = urlparse(self.path).path
        try:
            if path == '/api/health':
                return self._json({'status':'ready', 'mode':'synthetic_simulation'})
            if path == '/api/history':
                return self._json({'results':self.engine.repository.history()})
            if path.startswith('/api/jobs/'):
                return self._json(self.engine.status(path.split('/')[-1]))
            if path.startswith('/api/reports/'):
                return self._json(self.engine.repository.report(path.split('/')[-1]))
            if path.startswith('/api/'):
                return self._json({'error':'Endpoint not found.'},404)
            # Serve only the dedicated web directory, never models or the database.
            return super().do_GET()
        except ValueError as exc:
            return self._json({'error':str(exc)},404)

    def do_POST(self):
        if not self._local_request() or not self.headers.get('Content-Type','').startswith('application/json'):
            return self._json({'error':'Unsupported request.'},403)
        try:
            length = int(self.headers.get('Content-Length','0'))
            if not 0 < length <= 16384:
                raise ValueError('Request body is too large or empty.')
            data = json.loads(self.rfile.read(length))
            if not isinstance(data,dict):
                raise ValueError('Request body must be an object.')
            path = urlparse(self.path).path
            if path == '/api/initialize':
                return self._json(self.engine.initialize(data.get('inputs')),202)
            if path == '/api/save':
                self.engine.repository.save(data.get('prediction_id'))
                return self._json({'saved':True})
            return self._json({'error':'Endpoint not found.'},404)
        except (ValueError, TypeError) as exc:
            return self._json({'error':str(exc)},400)

    def end_headers(self):
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Referrer-Policy','no-referrer')
        self.send_header('Content-Security-Policy',
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
        super().end_headers()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data')
    parser.add_argument('--no-browser',action='store_true')
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(filename=args.data_dir/'app.log',level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    engine = Engine(args.data_dir)
    try:
        server = ThreadingHTTPServer(('127.0.0.1',args.port),partial(Handler,engine=engine))
    except OSError:
        engine.close()
        raise SystemExit(f'Port {args.port} is in use. Close the previous Casimir window or choose another port.')
    url = f'http://127.0.0.1:{args.port}/'
    print(f'Casimir is ready: {url}',flush=True)
    print('Keep this window open while using the app. Press Ctrl+C to close.',flush=True)
    if not args.no_browser:
        threading.Timer(.5,lambda:webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        engine.close()


if __name__ == '__main__':
    main()
