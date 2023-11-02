from __future__ import print_function
# This file is for tooling that's django-free, and so can be used anywhere
# (in particular, outside the web1 web application).
#
# This file should also avoid importing any other parts of the system,
# including PathHelper.  Tools for wrapping file databases in classes
# can live here if they don't require path information (e.g. the location
# of the file is passed in from the outside).  If these wrappers have
# path information built in, they can live in browse/files.py.

# XXX This should eventually be broken up and moved to dtk.  Since we
# XXX now mostly do imports at local scope instead of global, it's
# XXX possible for non-django code to call non-django functions that
# XXX share files with functions requiring django.
import os
import shutil
import errno
import itertools
from collections import Counter
import re
import six

# The following 2 classes make it easy for one process to communicate progress
# to another.  The process is broken into phases, and at the end of each phase
# the process writes a description for that phase.  Each phase can be written
# exactly once.  The sending and receiving end only need to agree on a filename
# to be used for passing the information.
#
# At the receiving end, the information is returned as two lists: a complete
# list and an incomplete list.  This makes it easy for the receiving process
# to stick a temporary status (maybe "in progress...", maybe a percentage
# completion if that can be estimated somehow) in the first element of the
# second list.
#
# Concatenating the two lists produces something the Django template engine
# can render very easily.
class ProgressWriter:
    def __init__(self,filename,headings):
        self.filename = filename
        self.headings = headings
        self.index = 0
        with open(self.filename,"w") as f:
            f.write('\t'.join(headings)+'\n')
    def put(self,heading,data):
        if self.index >= len(self.headings):
            raise AttributeError("Already full")
        if heading != self.headings[self.index]:
            raise AttributeError("Out of order")
        with open(self.filename,"a") as f:
            if self.index:
                f.write('\t')
            f.write(data)
        self.index += 1

class ProgressReader:
    def __init__(self,filename):
        self.filename = filename
    def get(self):
        if not os.path.exists(self.filename):
            return ([],[['Awaiting start','']])
        with open(self.filename,"r") as f:
            headings = next(f).strip('\n').split('\t')
            try:
                data = next(f).strip('\n').split('\t')
            except:
                data = []
        got = len(data)
        return ([list(x) for x in zip(headings,data)]
            ,[list(x) for x in zip(headings[got:],itertools.repeat(''))]
            )

def obfuscate(text):
    if not text:
        return ""
    min_length = 6
    max_length = 25
    # Instead of a random length, use a pseudo-random length based
    # on a hash of the drug name, so page rendering is consistent
    total = 0
    for ch in text:
        total += ord(ch)
    length = min_length + total % (max_length - min_length)
    #return text[0]+("*" * (length-1))
    return ("*" * length)

def sci_fmt(val):
    # return blank for blankish values
    if isinstance(val,str) and val.strip() == '':
        return ''
    if val is None:
        return ''
    # default is scientific notation
    val = float(val)
    exp = "%0.1e" % val
    # for values near 1, switch to floating point
    if exp[-3] in "+-":
        scale = int(exp[-3:])
        if -4 <= scale <= 4:
            # use 3 decimals by default, but more for small
            # numbers in the range above (so they don't show
            # up as 0)
            decimals = max(3,-scale)
            # tricky: fmt will look like "%.<val>f"
            fmt = "%%.%df" % decimals
            flt = fmt % val
            # strip trailing zeros
            flt = re.sub(r'0+$','',flt)
            flt = re.sub(r'\.$','',flt)
            return flt
    return exp

def make_sure_dir_exists(path):
    try:
        os.makedirs(path)
    except OSError as ex:
        if ex.errno != errno.EEXIST:
            raise

def copy_if_needed(from_dir, filename, to_dir):
    make_sure_dir_exists(to_dir)
    to_path = os.path.join(to_dir,filename)
    if not os.path.exists(to_path):
        from_path = os.path.join(from_dir,filename)
        shutil.copyfile(from_path,to_path)

def touch(fname):
    dirname=os.path.dirname(fname)
    if dirname != "":
        try:
            os.makedirs(dirname)
        except:
            pass
    with open(fname,'a'):
        os.utime(fname,None)

def strmatch(v1,v2):
    return str(v1).lower().strip() == str(v2).lower().strip()

def linecount(path):
    lines = 0
    try:
        with open(path,"r") as f:
            lines = len(f.readlines())
        print("file opened; got %d" % (lines,))
    except Exception as ex:
        print(repr(ex))
        print("got exception; %s" % (path, ))
        pass
    return lines

def percent(value):
    value = int(100*value)
    if value > 100:
        value = 100
    if value < 0:
        value = 0
    return str(value)+"%"

class Enum:
    """
    Common usages:
        EnumType = Enum([], [
                ('ENUM_VAL_1','Custom Label'),
                ('SECOND_TYPE',),
                ])
        enum_val = EnumType.SECOND_TYPE
        label = EnumType.get("label", enum_val)
        enum_choices = EnumType.choices()
    """
    def __init__(self,properties,members):
        self.properties = ['symbol','label','active']+properties
        self.members = []
        for i,v in enumerate(members):
            setattr(self,v[0],i)
            tmp = []
            for i,f in enumerate(self.properties):
                if i < len(v) and v[i] != None:
                    # if the field is specified, copy it
                    tmp.append(v[i])
                else:
                    # try to assign default
                    val = None
                    if strmatch('label',f):
                        val = v[0].title().replace('_',' ')
                    elif strmatch('active',f):
                        val = True
                    tmp.append(val)
            self.members.append(tuple(tmp))
    def get(self,pname,index,default=None):
        for i,n in enumerate(self.properties):
            if strmatch(n,pname):
                val = self.members[index][i]
                if val == None:
                    return default;
                return val;
        raise AttributeError("'"+pname+"' not found")
    def find(self,pname,value):
        for i,n in enumerate(self.properties):
            if strmatch(n,pname):
                for j,v in enumerate(self.members):
                    if strmatch(value,v[i]):
                        return j
                raise ValueError("'"+value+"' not found")
        raise AttributeError("'"+pname+"' not found")
    def choices(self):
        tmp = []
        for j,v in enumerate(self.members):
            if v[2]:
                tmp.append((j,v[1]))
        return tmp

class TripleMap:
    # {key1:{key2:[val,val,...],...},...}
    # This is a generic data structure for mapping two levels of keys
    # onto a list of values.
    def __init__(self):
        self.index = {}
    def add(self,key1,key2,value):
        idx2 = self.index.setdefault(key1,{})
        vlist = idx2.setdefault(key2,[])
        vlist.append(value)
    def find(self,key1,key2):
        if isinstance(key2,str):
            key2 = [key2]
        result = []
        if key1 in self.index:
            idx2 = self.index[key1]
            for k2 in key2:
                result += idx2.get(k2,[])
        return result
    def find_partial(self,key1,key2):
        # only key1 can be partial
        if isinstance(key2,str):
            key2 = [key2]
        result = []
        for k in self.index:
            try:
                if key1 in k:
                    idx2 = self.index[k]
                    for k2 in key2:
                        result += idx2.get(k2,[])
            except UnicodeDecodeError:
                pass
        return result
    def print_stats(self):
        # One common use case for this structure is to associate key1
        # to an ideally unique val by means of multiple types of evidence,
        # where the type of evidence is key2.  This method prints out
        # stats related to the number of mappings, the number of
        # inconsistencies, and the prevalence of various evidence
        # combinations.
        print()
        print('key1 count',len(self.index))
        uniques = 0
        pattern_counts = Counter()
        for key1,idx2 in six.iteritems(self.index):
            if len(idx2) == 1:
                uniques += 1
            else:
                print(key1,idx2)
            for v in idx2.values():
                s = set()
                for item in v:
                    s.add(item)
                p = ",".join(sorted(list(s)))
                pattern_counts[p] += 1
        print(uniques,"key1 values mapped uniquely")
        print('evidence combinations:',pattern_counts)

