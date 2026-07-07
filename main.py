import logging
import logging.config

class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.args and len(record.args) >= 3 and record.args[2] != "/health"

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        },
        "system_formatter": {
            "format": "%(asctime)s [%(levelname)s] System: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        },
        "routes_formatter": {
            "format": "%(asctime)s [%(levelname)s] Routes: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        },
    },
    "handlers": {
        "default": {
            "formatter": "standard",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
        "system_handler": {
            "formatter": "system_formatter",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
        "routes_handler": {
            "formatter": "routes_formatter",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "API": {"handlers": ["default"], "level": "INFO"},
        "Engine": {"handlers": ["default"], "level": "INFO"},
        "uvicorn": {"handlers": ["default"], "level": "INFO"},
        "uvicorn.error": {
            "handlers": ["system_handler"],
            "level": "INFO",
            "propagate": False
        },
        "uvicorn.access": {"handlers": ["routes_handler"], "level": "INFO", "propagate": False},
    },
}

logging.config.dictConfig(LOGGING_CONFIG)
logging.getLogger("uvicorn.access").addFilter(EndpointFilter())
logger = logging.getLogger("API")

from engine import config
from gateway.app import create_app

app = create_app(config)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT, log_config=LOGGING_CONFIG)