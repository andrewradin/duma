from django.shortcuts import render

from django.contrib.auth.decorators import login_required
from notes.models import Note

# Create your views here.
@login_required
def history(request,note_id):
    from browse.views import is_demo
    if is_demo(request.user):
        n = Note(label='Notes cannot be accessed in demo mode')
    else:
        n = Note.objects.get(pk=note_id)
    prev = None
    deltas = []
    from dtk.text import diffstr
    for vers in n.get_history(request.user.username):
        if prev is None:
            diff = [ ('','',vers.text) ]
        else:
            diff = diffstr(prev.text,vers.text,keep=20,minmatch=3)
        deltas.append( (vers,diff) )
        prev = vers
    return render(request
                ,'notes/history.html'
                ,{ 'deltas':reversed(deltas)
                 , 'note':n
                 , 'page_tab':n.label
                 }
                )
