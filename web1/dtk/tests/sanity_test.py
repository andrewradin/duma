

def test_sanity():
    """Just tests that we can import a couple of things."""
    import path_helper
    import aws_op
    assert True == True

def test_django_sanity():
    import django
    import django_setup

def test_django_model_imports():
    from browse.models import Workspace
    from drugs.models import Drug
    from runner.models import Process
