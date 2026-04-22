"""server.py — Minimal web viewer for eabrain memory."""

import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

from memory import MemoryDB


_WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


class EabrainHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _json_response(self, data, status=200):
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _get_db(self):
        return MemoryDB(os.path.join(self.server.cfg["eabrain_dir"], "memory.db"))

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/":
            self._serve_html()
        elif path == "/api/stats":
            self._api_stats()
        elif path == "/api/timeline":
            self._api_timeline(params)
        elif path == "/api/search":
            self._api_search(params)
        elif path.startswith("/api/observations/"):
            obs_id = path.split("/")[-1]
            self._api_observation(obs_id)
        else:
            self.send_error(404)

    def _serve_html(self):
        html_path = os.path.join(_WEB_DIR, "viewer.html")
        if not os.path.exists(html_path):
            self.send_error(404, "viewer.html not found")
            return
        with open(html_path, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _api_stats(self):
        db = self._get_db()
        data = db.stats()
        db.close()
        self._json_response(data)

    def _api_timeline(self, params):
        db = self._get_db()
        project = params.get("project", [None])[0]
        limit = int(params.get("last", [20])[0])
        since = params.get("since", [None])[0]
        data = db.timeline(project=project, limit=limit, since=since)
        for entry in data:
            for o in entry["observations"]:
                o.pop("embedding", None)
        db.close()
        self._json_response(data)

    def _api_search(self, params):
        db = self._get_db()
        q = params.get("q", [""])[0]
        results = db.simd_search(q, limit=20)
        for r in results:
            r.pop("embedding", None)
        db.close()
        self._json_response({"observations": results})

    def _api_observation(self, obs_id):
        db = self._get_db()
        row = db.conn.execute("SELECT * FROM observations WHERE id = ?", (obs_id,)).fetchone()
        if not row:
            db.close()
            self._json_response({"error": "not found"}, status=404)
            return
        data = dict(row)
        data.pop("embedding", None)
        db.close()
        self._json_response(data)


def make_server(cfg: dict, port: int = 37777) -> HTTPServer:
    server = HTTPServer(("127.0.0.1", port), EabrainHandler)
    server.cfg = cfg
    return server


def serve(cfg: dict, port: int = 37777):
    server = make_server(cfg, port)
    print(f"eabrain viewer: http://localhost:{port}")
    server.serve_forever()
