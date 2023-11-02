#!/usr/bin/env python

import sys
sys.path.insert(1,"../../web1")
import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web1.settings")
import django
django.setup()

# This code dumps a collection, re-creating the import file

from drugs.models import Collection,Drug,Prop

# the encode() isn't necessary when outputting to a terminal (where LOCALE
# probably does the right thing), but when piped to another process, or
# redirected to a file, it seems to choose C encoding, and dies on unicode.
## def outrow(f,l): f.write("\t".join(l)+"\n")
def outrow(f,l): f.write( ("\t".join(l)+"\n").encode('utf8') )

def dump(coll_id,outf,clean):
    coll = Collection.objects.get(id=coll_id)
    outrow(outf,[coll.key_name,'attribute','value'])
    col_qs = Drug.objects.filter(collection_id=coll_id)
    props = [Prop.get(Prop.NAME)] + coll.static_properties_list()
    for drug in col_qs:
        key_val = getattr(drug,coll.key_name)
        for prop in props:
            vals = getattr(drug,prop.name+'_set')
            for val in vals:
                if clean:
                    if val == 'None':
                        continue
                    try:
                        val = val.strip()
                    except AttributeError:
                        pass
                outrow(outf,[
                    key_val,
                    prop.name,
                    unicode(val),
                    ])

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='extract collection from db')
    parser.add_argument('--clean',action='store_true')
    parser.add_argument('collection_id')
    args = parser.parse_args()

    dump(args.collection_id,sys.stdout,args.clean)
