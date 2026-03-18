#!/usr/bin/env python3
"""
Simple HTTP server to view the generated HTML site.
"""

import http.server
import socketserver
import webbrowser
from pathlib import Path

PORT = 8000
DIRECTORY = "site"

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

def serve():
    """Start a local web server to view the site."""
    site_path = Path(DIRECTORY)
    if not site_path.exists():
        print(f"Error: Site directory '{DIRECTORY}' does not exist.")
        print("Run generate_html_site.py first to create the site.")
        return

    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        url = f"http://localhost:{PORT}"
        print(f"Serving site at: {url}")
        print(f"Directory: {site_path.absolute()}")
        print(f"\nPress Ctrl+C to stop the server")

        # Optionally open browser
        # webbrowser.open(url)

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")

if __name__ == "__main__":
    serve()
