from django.db import models

import logging
logger = logging.getLogger(__name__)
from django.db import transaction

# TL;DR Adding a note to a model:
#
# in the holder model:
# class SomeModel(models.Model):
#   # add a holder attribute
#   example_note=models.ForeignKey(Note,null=True,blank=True,
#                                  on_delete=models.CASCADE)
#   # add a get method
#   def get_example_note_text(self):
#       return Note.get(self,'example_note','')
#   # ... or for potentially private notes:
#   def get_example_note_text(self,user):
#       return Note.get(self,'example_note',user)
#   # add a note_info method (or add another if-case to the existing method)
#   # that defines the label which will identify the note when it appears in
#   # a generic note view; it should include the type of note and some
#   # instance identification.
#   def note_info(self,attr):
#       if attr == 'example_note':
#           return {
#                  'label':'example note for SomeModel %d'%self.id,
#                  }
#       raise Exception("bad note attr '%s'" % attr)
#
# and in view code where note is set/updated, call:
#   Note.set(some_model,'example_note',user,text,private=True or False)

class Note(models.Model):
    # note handling is generic, and the source of any note isn't
    # tracked at this level; the label field allows us to label
    # history displays reasonably, and may help with debugging
    label = models.CharField(max_length=250,blank=True,default="")
    # if set, access is restricted to the specified user
    private_to = models.CharField(max_length=50,blank=True,default="")

    # internals
    def _latest_version(self):
        return self.noteversion_set.order_by('-id')[0]
    def _check_access(self,user):
        self._pre_check_access(self.__dict__,user)
    @classmethod
    def _pre_check_access(cls,opts,user):
        must_match = opts.get('private_to','')
        if must_match:
            err = ''
            if not user:
                err = 'note private to %s; accessor unspecified' % (
                        must_match,
                        )
            elif must_match != user:
                err = 'note private to "%s"; denying access to "%s"' % (
                        must_match,
                        user,
                        )
            if err:
                logger.warning(err)
                raise Exception(err)
    # The get() and set() class methods are the preferred method of
    # accessing notes.  They're based on the assumption that another
    # record object holds the note id, which is usually null.  These
    # methods:
    # - return an empty text for null note ids
    # - store the note id on first write
    # - enforce access control
    #
    # To simplify retrieving note text in templates, models can have
    # get methods that call Note.get(), supplying all the needed arguments.
    #
    # Because the Note is really created on first write, rather than
    # at the same time as the holder, the holder must implement a
    # note_info() method that takes the attribute name and returns
    # a hash containing any initial values for note-level attributes.
    @classmethod
    def get(cls,holder,attr,user):
        # user is a string, and can be '' unless you're accessing
        # a potentially private message
        root = getattr(holder,attr)
        if not root:
            return ''
        return root.get_latest(user)
    
    def get_latest(self, user=''):
        self._check_access(user)
        ver = self._latest_version()
        return ver.text

    @classmethod
    def set(cls,holder,attr,user,text,private=False):
        with transaction.atomic():
            # user is a string, and is required to properly record audit trails
            text = text.strip()
            root = getattr(holder,attr)
            if not root:
                if not text:
                    return
                # add new note
                opts = holder.note_info(attr)
                if private:
                    assert user
                    opts['private_to'] = user
                cls._pre_check_access(opts,user)
                root = cls(**opts)
                root.save()
                setattr(holder,attr,root)
                holder.save() # XXX make optional?
            else:
                root._check_access(user)
                # get existing note; return if text unchanged
                cur = root._latest_version()
                if text == cur.text:
                    return
            # save new version of text, if we haven't returned by now
            ver = NoteVersion(version_of=root,text=text,created_by=user)
            ver.save()
    # access to version history
    # XXX maybe this should copy info into a list of dicts, so that
    # XXX clients never see NoteVersion objects at all
    def get_history(self,user=None):
        self._check_access(user)
        return self.noteversion_set.order_by('id')
    @classmethod
    def batch_note_lookup(cls,note_ids,user):
        '''Yields (note_id,text) pairs for all note_ids available to user.
        '''
        from django.db.models import Max
        version_qs = cls.objects.filter(
                id__in=note_ids,
                private_to__in=('',user),
                ).annotate(Max('noteversion__id'))
        vers_ids=set(version_qs.values_list('noteversion__id__max',flat=True))
        text_qs = NoteVersion.objects.filter(id__in=vers_ids)
        for note_id,text in text_qs.values_list('version_of_id','text'):
            yield note_id,text

class NoteVersion(models.Model):
    version_of = models.ForeignKey(Note, on_delete=models.CASCADE)
    text = models.TextField()
    created_by = models.CharField(max_length=50,blank=True,default="")
    created_on = models.DateTimeField(auto_now_add=True)

    class Meta:
        index_together = [
                ['created_by'],
                ]

