
def get_needs_vote_icon(wsa, user):
    from browse.models import Vote
    from dtk.html import tag_wrap, glyph_icon
    if Vote.needs_this_vote(wsa,user):
        vote_icon = tag_wrap(
                    'font',
                    glyph_icon('arrow-right'),
                    attr={'color':'firebrick'},
                    )
    else:
        vote_icon=''
    return vote_icon

def prioritize_demerits():
    order = ['Patented', 'Non-novel class',
            'Unavailable', 'Exacerbating',
            'Modality', 'Tox',
            'Non-unique', 'No MOA',
            'Data Quality', 'Ubiquitous',
           ]

    from browse.models import Demerit
    all_demerits = Demerit.objects.all().values_list('desc', flat=True)
    for demerit in all_demerits:
        if demerit not in order:
            order.append(demerit)
    return order
