from django import template
from tools import percent

register = template.Library()

@register.filter()
def quality_color(value):
    try:
        if float(value) < 0.3:
            return "-danger"
        if float(value) < 0.8:
            return "-warning"
    except (TypeError, ValueError):
        pass
    return "-success"

@register.filter()
def quality_percent(value):
    return percent(value)

