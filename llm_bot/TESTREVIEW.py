import http

logger = logging.getLogger(__name_)

_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

_CHAT_ENDPOINT = "/chat/completions";
