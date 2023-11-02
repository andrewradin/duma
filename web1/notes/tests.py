from django.test import TestCase

from notes.models import Note

# Create your tests here.
class LifeCycleTestCase(TestCase):
    def test_simple(self):
        t1="this is some text"
        u1="me"
        l1="the default label"
        class DummyHolder:
            def note_info(self,attr):
                return {'label':l1}
            def __init__(self):
                self.my_note=None
            def save(self):
                pass
            def get_text(self):
                return Note.get(self,'my_note','')
        h = DummyHolder()
        Note.set(h,'my_note',u1,t1)
        self.assertEqual(h.get_text(),t1)
        t2="some different text"
        u2="somebody else"
        Note.set(h,'my_note',u2,t2)
        self.assertEqual(h.get_text(),t2)
        t3="a third version"
        Note.set(h,'my_note',u1,t3)
        self.assertEqual(h.get_text(),t3)
        # verify history works
        history=[x for x in h.my_note.get_history()]
        self.assertEqual(history[0].text,t1)
        self.assertEqual(history[0].created_by,u1)
        self.assertEqual(history[1].text,t2)
        self.assertEqual(history[1].created_by,u2)
        self.assertEqual(history[2].text,t3)
        self.assertEqual(history[2].created_by,u1)
        # verify unchanged text doesn't get written
        Note.set(h,'my_note',u1,t3)
        self.assertEqual(h.my_note.get_history().count(),3)
        # and, this happens even if the user changes
        Note.set(h,'my_note',u2,t3)
        self.assertEqual(h.my_note.get_history().count(),3)

