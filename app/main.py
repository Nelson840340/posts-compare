"""入口：python -m app.main"""
import logging
import os

import uvicorn

from app.config import load_config
from app.service import build_app

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")

cfg = load_config()
app = build_app(cfg)

if __name__ == "__main__":
    uvicorn.run(app, host=os.environ.get("SIM_HOST", "127.0.0.1"),
                port=int(os.environ.get("SIM_PORT", "8000")))
