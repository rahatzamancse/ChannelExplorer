import functools
import hashlib
import json
import redis

# Connect to Redis server
redis_client = redis.Redis(host='localhost', port=6379, db=0)

def redis_cache(ttl=3600):
    def decorator_cache(func):
        @functools.wraps(func)
        def wrapper_cache(*args, **kwargs):
            # Create a unique key based on the function name and arguments
            # only pick args and kwargs that are not private
            arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]
            key_parts = [func.__name__] \
                + [arg for arg_name, arg in zip(arg_names, args) if not arg_name.startswith("_")] \
                + [f"{k}={v}" for k, v in kwargs.items() if not k.startswith("_")]
            key = hashlib.sha256(json.dumps(key_parts, sort_keys=True).encode()).hexdigest()

            # Try retrieving the cache
            cached_result = redis_client.get(key)
            if cached_result:
                print("Returning result from cache")
                return json.loads(cached_result)

            # Call the function and cache its result
            result = func(*args, **kwargs)
            redis_client.setex(key, ttl, json.dumps(result))
            return result

        return wrapper_cache
    return decorator_cache