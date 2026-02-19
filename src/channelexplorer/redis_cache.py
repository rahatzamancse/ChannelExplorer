import functools
import hashlib
import json
import logging
import os
import redis

logger = logging.getLogger(__name__)

_fallback_cache: dict[str, str] = {}

REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.environ.get('REDIS_PORT', '6379'))
REDIS_DB = int(os.environ.get('REDIS_DB', '0'))

try:
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
    redis_client.ping()
except redis.ConnectionError:
    logger.warning("Redis unavailable, using in-memory fallback cache")
    redis_client = None


class _FallbackClient:
    """Minimal dict-backed fallback when Redis is unavailable."""
    def get(self, key):
        return _fallback_cache.get(key)

    def set(self, key, value):
        _fallback_cache[key] = value if isinstance(value, bytes) else str(value).encode()

    def setex(self, key, ttl, value):
        self.set(key, value)

    def flushdb(self):
        _fallback_cache.clear()

    def ping(self):
        return True


if redis_client is None:
    redis_client = _FallbackClient()


def redis_cache(ttl=3600):
    def decorator_cache(func):
        @functools.wraps(func)
        def wrapper_cache(*args, **kwargs):
            arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]
            key_parts = [func.__name__] \
                + [arg for arg_name, arg in zip(arg_names, args) if not arg_name.startswith("_")] \
                + [f"{k}={v}" for k, v in kwargs.items() if not k.startswith("_")]
            key = hashlib.sha256(json.dumps(key_parts, sort_keys=True).encode()).hexdigest()

            try:
                cached_result = redis_client.get(key)
                if cached_result:
                    return json.loads(cached_result)
            except Exception:
                pass

            result = func(*args, **kwargs)
            try:
                redis_client.setex(key, ttl, json.dumps(result))
            except Exception:
                pass
            return result

        return wrapper_cache
    return decorator_cache
