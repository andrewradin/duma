


from dtk.cache import cached_dict_elements
import pytest

@pytest.fixture(autouse=True)
def mem_cache(settings):
    print("Setting up mem cache; was", settings.CACHES['cached_dict_elements'])
    # For any tests in this file, we need a real cache setup.
    settings.CACHES = {
        'cached_dict_elements': {'BACKEND': 'django.core.cache.backends.locmem.LocMemCache'},
        'dummy': {'BACKEND': 'django.core.cache.backends.dummy.DummyCache'},
        'mem': {'BACKEND': 'django.core.cache.backends.locmem.LocMemCache'},
        }
    yield None

    # Clear caches between tests.
    from django.core.cache import caches
    caches['mem'].clear()
    caches['cached_dict_elements'].clear()

    print("Tearing down mem cache")

def test_dict_cache():

    static_data = {
            1: {
                'foo': 'bar',
                'protein': 'gene'
            },

            2: {
                'foo': 'notbar'
            }
        }


    def lu_converter(version, keys):
        if version is None:
            return False, None, None, None
        else:
            return True, version, keys, 'keys'

    lookup_log = []

    @cached_dict_elements(lu_converter)
    def lookup_version(version, keys):
        lookup_log.append((version, keys))
        if version is None:
            return "Unversioned!"

        return {k: static_data[version][k] for k in keys}


    assert len(lookup_log) == 0

    assert lookup_version(1, keys=['foo']) == {'foo': 'bar'}
    assert len(lookup_log) == 1
    assert lookup_log.pop()  == (1, ['foo']), "Normal uncached lookup"

    assert lookup_version(1, ['foo']) == {'foo': 'bar'}
    assert len(lookup_log) == 0, "Cached lookup, nothing should have changed"


    assert lookup_version(1, keys=['protein', 'foo']) == {'protein': 'gene', 'foo': 'bar'}
    assert len(lookup_log) == 1
    assert lookup_log.pop() == (1, ['protein']), "Only queried the new key"


    assert lookup_version(2, keys=['foo']) == {'foo': 'notbar'}, "Different version"
    assert len(lookup_log) == 1
    lookup_log.pop()

    assert lookup_version(None, keys=['foo']) == "Unversioned!"
    assert len(lookup_log) == 1
    lookup_log.pop()

    assert lookup_version(None, keys=['foo']) == "Unversioned!"
    assert len(lookup_log) == 1
    lookup_log.pop()


def test_dict_cache_list():

    static_data = {
            1: {
                'foo': 'bar',
                'protein': 'gene'
            },

            2: {
                'foo': 'notbar'
            }
        }


    def lu_converter(version, keys):
        if version is None:
            return False, None, None, None
        else:
            return True, version, keys, 'keys'

    lookup_log = []

    @cached_dict_elements(lu_converter, list_style=True)
    def lookup_version(version, keys):
        lookup_log.append((version, keys))
        if version is None:
            return "Unversioned!"

        return [static_data[version][k] for k in keys]


    assert len(lookup_log) == 0

    assert lookup_version(1, keys=['foo']) == ['bar']
    assert len(lookup_log) == 1
    assert lookup_log.pop()  == (1, ['foo']), "Normal uncached lookup"

    assert lookup_version(1, keys=['foo']) == ['bar']
    assert len(lookup_log) == 0, "Cached lookup, nothing should have changed"


    assert lookup_version(1, ['protein', 'foo']) == ['gene', 'bar']
    assert len(lookup_log) == 1
    assert lookup_log.pop() == (1, ['protein']), "Only queried the new key"


    assert lookup_version(2, keys=['foo']) == ['notbar'], "Different version"
    assert len(lookup_log) == 1
    lookup_log.pop()

    assert lookup_version(None, keys=['foo']) == "Unversioned!"
    assert len(lookup_log) == 1
    lookup_log.pop()

    assert lookup_version(None, keys=['foo']) == "Unversioned!"
    assert len(lookup_log) == 1
    lookup_log.pop()


def test_dict_cache_mm():

    static_data = {
            1: [
                ('foo', 'bar'),
                ('protein', 'gene'),
                ('foo', 'bar2'),
            ],

            2: {
                ('foo', 'notbar'),
            }
        }


    def lu_converter(version, keys):
        if version is None:
            return False, None, None, None
        else:
            return True, version, keys, 'keys'

    lookup_log = []

    @cached_dict_elements(lu_converter, multimap_style=True)
    def lookup_version(version, keys):
        lookup_log.append((version, keys))
        if version is None:
            return "Unversioned!"

        from dtk.data import MultiMap
        return MultiMap((k, v) for (k, v) in static_data[version] if k in keys)


    assert len(lookup_log) == 0

    assert lookup_version(1, keys=['foo']).fwd_map() == {'foo': {'bar', 'bar2'}}
    assert len(lookup_log) == 1
    assert lookup_log.pop()  == (1, ['foo']), "Normal uncached lookup"

    assert lookup_version(1, ['foo']).fwd_map() == {'foo': {'bar', 'bar2'}}
    assert len(lookup_log) == 0, "Cached lookup, nothing should have changed"


    assert lookup_version(1, keys=['protein', 'foo']).fwd_map() == {'protein': {'gene'}, 'foo': {'bar', 'bar2'}}
    assert len(lookup_log) == 1
    assert lookup_log.pop() == (1, ['protein']), "Only queried the new key"


    assert lookup_version(2, keys=['foo']).fwd_map() == {'foo': {'notbar'}}, "Different version"
    assert len(lookup_log) == 1
    lookup_log.pop()

    assert lookup_version(None, keys=['foo']) == "Unversioned!"
    assert len(lookup_log) == 1
    lookup_log.pop()

    assert lookup_version(None, keys=['foo']) == "Unversioned!"
    assert len(lookup_log) == 1
    lookup_log.pop()



# Test caching behaviors

# Dummy caches don't store values at all, even when set.
def test_dummy_cache():
    from django.core.cache import caches
    assert caches['dummy'].get('key1', None) is None
    assert caches['dummy'].get('key2', None) is None
    caches['dummy'].set('key1', 'value1')
    assert caches['dummy'].get('key1', None) == None


# Mem caches should set values, but they should not persist across tests.
# NOTE: This depends on the fixture cache clearing at the top of this file.
def test_nonshared_cache1():
    from django.core.cache import caches
    assert caches['mem'].get('key1', None) is None
    assert caches['mem'].get('key2', None) is None
    caches['mem'].set('key1', 'value1')
    assert caches['mem'].get('key1', None) == 'value1'
def test_nonshared_cache2():
    from django.core.cache import caches
    assert caches['mem'].get('key1', None) is None
    assert caches['mem'].get('key2', None) is None
    caches['mem'].set('key2', 'value2')
    assert caches['mem'].get('key2', None) == 'value2'