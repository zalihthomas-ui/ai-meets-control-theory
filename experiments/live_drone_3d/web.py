#!/usr/bin/env python3
"""Web launcher for the 3D Crazyflie 2.0 WebGL sandbox.

Serves the self-contained Three.js visualizer on local port and opens the browser.

Usage:
    python experiments/live_drone_3d/web.py
    python experiments/live_drone_3d/web.py --port 8080 --no-browser
"""

from __future__ import annotations

import argparse
import http.server
import socketserver
import threading
import webbrowser
from pathlib import Path


def serve(port: int = 8000, open_browser: bool = True):
    here = Path(__file__).resolve().parent
    index_file = here / "index.html"
    if not index_file.is_file():
        raise FileNotFoundError(f"Missing {index_file}")

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(here), **kwargs)

        def log_message(self, format, *args):
            pass  # quiet output

    with socketserver.TCPServer(("", port), Handler) as httpd:
        url = f"http://localhost:{port}/index.html"
        print(f"==================================================")
        print(f" 3D Crazyflie 2.0 WebGL Live Sandbox")
        print(f" Serving at: {url}")
        print(f" Press Ctrl+C to stop the server")
        print(f"==================================================")

        if open_browser:
            threading.Timer(0.5, lambda: webbrowser.open(url)).start()

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped server.")


def main():
    parser = argparse.ArgumentParser(description="Serve the 3D drone sandbox")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")
    args = parser.parse_args()
    serve(port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
