import pytest
from dtk.lazy_loader import LazyLoader

class ExampleClass(LazyLoader):
    _kwargs = ['valid_keyword_arg']
    def _lazy_member_loader(self):
        self.execution_count += 1
        return 'dynamic_value'

def test_lazy_loader():
    # verify instantiation options
    ec = ExampleClass()
    with pytest.raises(TypeError):
        ec = ExampleClass(invalid_keyword_arg = 'some_initial_value')
    ec = ExampleClass(valid_keyword_arg = 'my_initial_value')
    assert ec.valid_keyword_arg == 'my_initial_value'
    # test lazy member loading
    ec.execution_count = 0
    assert ec.lazy_member == 'dynamic_value'
    assert ec.execution_count == 1
    # test that subsequent references don't re-execute the loader
    assert ec.lazy_member == 'dynamic_value'
    assert ec.execution_count == 1
    # test that pre-loading a member manually suppresses loader execution
    ec = ExampleClass()
    ec.execution_count = 0
    ec.lazy_member = 'preset_value'
    assert ec.lazy_member == 'preset_value'
    assert ec.execution_count == 0
