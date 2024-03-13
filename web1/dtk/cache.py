import logging
logger = logging.getLogger(__name__)
class Cacher:
    '''A wrapper for Django caching.

    This class is targeted at caching expensive intermediate results.
    The check() method takes a key which identifies a result in the cache,
    and a function to be executed to calculate the result if the cache
    entry isn't present.  If the function is executed, a new cache entry
    is created as well.

    The 'prefix' should be a dot-separated string that identifies the
    hash client (<module>.<class> or <module>.<function>).

    The 'func' may be a lambda that calls some other function with a
    specific set of parameters.  The 'key' passed with the function
    should identify that function (within the scope of all calls that
    share a 'prefix') and, if necessary, that function's parameters.
    The signature() method provides an easy way to turn any python
    object representing the function's parameters into a string.
    '''
    @staticmethod
    def signature(obj):
        from pprint import pformat
        from hashlib import md5
        return md5(pformat(obj).encode('utf8')).hexdigest()
    def __init__(self,prefix,log_miss=True):
        self._prefix=prefix
        self._log_miss=log_miss
    def _get_cache(self):
        try:
            from django.core.cache import caches
        except Exception as ex:
            # if django isn't set up, the above import throws an exception;
            # the actual exception type isn't available either (it's
            # django.core.exceptions.ImproperlyConfigured), so just look at
            # the exception message, and fall back to non-cached behavior if
            # this is what happened.
            if 'DJANGO_SETTINGS_MODULE' in str(ex):
                return None
            raise
        from django.core.cache import InvalidCacheBackendError
        try:
            return caches[self._prefix]
        except InvalidCacheBackendError:
            return caches['default']
    def check(self,key,func):
        cache = self._get_cache()
        if cache is None:
            return func()
        val = cache.get(self._prefix+'.'+key)
        if val is None:
            import datetime
            start = datetime.datetime.now()
            val = func()
            end = datetime.datetime.now()
            cache.set(self._prefix+'.'+key,val)
            if self._log_miss:
                logger.info("prefix '%s' key '%s' cache miss; took %s",
                            self._prefix,
                            key,
                            str(end-start),
                            )
        else:
            logger.debug("prefix '%s' key '%s' cache hit",self._prefix,key)
        return val
    def delete_many(self,key_list):
        cache = self._get_cache()
        if cache is None:
            return
        cache.delete_many([
                self._prefix+'.'+key
                for key in key_list
                ])

def frozenargs(*args, **kwargs):
    return tuple(list(args) + sorted(kwargs.items())),

def cached(version=0, argsfunc=frozenargs):
    """A basic caching decorator.

    This works fine if your output is a pure function of your inputs,
    all outputs are pickleable, and inputs are usable as a dict key.

    argsfunc outputs the key that it gets cached on; by default it is
    just a tuple of the arguments + sorted kwargs
    """
    def dec(func):
        from functools import wraps
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = (func.__qualname__, argsfunc(*args, **kwargs), version)
            from django.core.cache import caches
            cache = caches['cached_dict_elements']

            out = cache.get(key, default=None)
            if out is not None:
                return out
            out = func(*args, **kwargs)
            cache.set(key, out)

            return out
        return wrapper
    return dec


def cached_dict_elements(convert_func, list_style=False, multimap_style=False):
    """Decorator for caching bulk-lookup/compute functions on a per-element basis.

    Given a function that returns some value for each of a set of inputs, this allows
    each input/result to be separately cached.  On subsequent calls, only the subset of
    inputs that have not previously been computed need to be re-computed.

    The convert_func returns details about the function being cached, allowing
    fine-grained control.  It should take the same arguments as the decorated function
    and return a tuple (do_cache, cache_key_prefix, keys, key_argument_name).

    By default it assumes output is of the style {key: result} for each key.  If instead
    the function returns [results], each corresponding to the input [keys] array, use 'list_style=True'.

    See dtk/tests/cache_test for examples.
    """
    def _wrapper(func_to_cache):
        import inspect
        args_names = inspect.getfullargspec(func_to_cache)[0]
        args_idxs = {k:i for i, k in enumerate(args_names)}
        # Copy docstrings & names & such
        from functools import wraps
        @wraps(func_to_cache)
        def _inner_wrapper(*args, **kwargs):
            # It would be marginally more efficient to do this in the outer wrapper,
            # but that causes a django dependency to get inserted at definition rather
            # than runtime, which would require a bunch of django_setups in non-django scripts.
            from django.core.cache import caches
            cache = caches['cached_dict_elements']
            data = convert_func(*args, **kwargs)
            if data is False or data[0] is False:
                return func_to_cache(*args, **kwargs)
            do_cache, cache_static_key, element_keys, key_fieldname = data
            cache_static_key = (func_to_cache.__qualname__, cache_static_key)


            # Find which of element keys are already cached.
            cached_results = {}
            uncached_keys = []
            cache_keys = ((cache_static_key, k) for k in element_keys)
            cached_results = cache.get_many(cache_keys)
            cached_results = {k[-1]: v for k, v in cached_results.items()}

            uncached_keys = list(set(element_keys) - cached_results.keys())

            if uncached_keys:
                if key_fieldname in kwargs:
                    kwargs = {**kwargs, key_fieldname: uncached_keys}
                else:
                    idx = args_idxs[key_fieldname]
                    assert len(args) > idx, f"Key parm {key_fieldname} not a keyword argument, and not {idx} args"
                    args = [uncached_keys if i == idx else args[i] for i in range(len(args))]

                new_results = func_to_cache(*args, **kwargs)

                # Update new_results into cache.
                if list_style:
                    new_results = dict(zip(uncached_keys, new_results))
                elif multimap_style:
                    new_results = new_results.fwd_map()

                new_cacheresults = {(cache_static_key, k):v for k,v in new_results.items()}
                cache.set_many(new_cacheresults)
                cached_results.update(new_results)

            if list_style:
                return [cached_results.get(k) for k in element_keys]
            elif multimap_style:
                from dtk.data import MultiMap
                return MultiMap.from_fwd_map(cached_results)
            else:
                return cached_results
        return _inner_wrapper
    return _wrapper
