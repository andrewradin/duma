
def filter_in_and_order(qs, key_name, objs):
    """Returns a queryset that is filtered to key_name__in=objs, in the same orer as objs.

    Inspired by https://stackoverflow.com/a/37648265

    This constructs a fairly awful SQL query, so it's not clear that it's better than reordering
    in python, but could switch to that if it becomes problematic.
    """
    from django.db.models import Case, When
    order = Case(*[When(**{key_name:key_val, 'then':pos}) for pos, key_val in enumerate(objs)])
    filter_args = {key_name + '__in': objs}
    return qs.filter(**filter_args).order_by(order)
