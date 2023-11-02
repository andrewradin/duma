from django import forms

class VoteForm(forms.ModelForm):
    note = forms.CharField(
            widget=forms.Textarea(attrs={'rows':'4','cols':'60'}),
            required=False,
            )
    class Meta:
        from browse.models import Vote
        model = Vote
        fields = ['recommended']
    def __init__(self, user, *args, **kwargs):
        super(VoteForm,self).__init__(*args, **kwargs)
        from notes.models import Note
        self.initial['note'] = Note.get(
                                        self.instance,
                                        'note',
                                        user=user.username,
                                        )

