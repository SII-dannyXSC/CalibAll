"""Flask application factory for CalibAll web interface."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Optional

from flask import Flask

from caliball.web.state import SharedState


def create_app(
    shared_state: Optional[SharedState] = None,
    template_folder: Optional[str] = None,
) -> Flask:
    """Create and configure the Flask application.

    Args:
        shared_state: Thread-safe shared state. Created if None.
        template_folder: Path to templates directory.
    """
    if template_folder is None:
        template_folder = str(Path(__file__).parent / "templates")

    app = Flask(__name__, template_folder=template_folder)
    app.config["SECRET_KEY"] = os.urandom(24)
    app.config["JSON_AS_ASCII"] = False

    # Shared state
    if shared_state is None:
        shared_state = SharedState()
    app.config["shared_state"] = shared_state

    # Register blueprints
    from caliball.web.routes.config_routes import config_bp
    from caliball.web.routes.annotate_routes import annotate_bp
    from caliball.web.routes.pipeline_routes import pipeline_bp

    app.register_blueprint(config_bp)
    app.register_blueprint(annotate_bp)
    app.register_blueprint(pipeline_bp, url_prefix="/api/pipeline")

    return app


def run_app(
    app: Flask,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
) -> threading.Thread:
    """Run Flask app in a background thread.

    Returns the server thread.
    """
    th = threading.Thread(
        target=lambda: app.run(host=host, port=port, threaded=True, use_reloader=False),
        daemon=True,
    )
    th.start()

    url = f"http://{host}:{port}/"
    print(f"[web] 服务已启动: {url}")
    if open_browser:
        import webbrowser
        webbrowser.open(url)

    return th
