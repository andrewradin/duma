from django import template
from django.utils.safestring import mark_safe
import os
from path_helper import PathHelper

register = template.Library()

@register.filter()
def prot2gene(prot):
    from browse.models import Protein
    ret = Protein.get_gene_of_uniprot(prot)
    if not ret:
        ret = "("+prot+")"
    return ret

@register.simple_tag(takes_context=True)
def quality_links(context,tissue):
    return tissue.quality_links(context['job_cross_checker'])
