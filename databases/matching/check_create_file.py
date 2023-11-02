#!/usr/bin/env python3

import django_setup
import os

def is_utf8(s):
    try:
        # With python3 we're doing the unicode decode on load, so it is unlikely that we will
        # ever fail here. Still, might as well check that we can correctly output as utf8 here.
        s.encode('utf8')
        return True
    except UnicodeEncodeError:
        return False

def is_int(s):
    try:
        int(s)
        return True
    except TypeError:
        return False

def is_float(s):
    try:
        float(s)
        return True
    except TypeError:
        return False

def verify(path):
    # parse and verify filename format
    basename = path.split('/')[-1]
    from dtk.files import VersionedFileName
    file_class=basename.split('.')[0]
    vfn = VersionedFileName(file_class=file_class,name=basename)
    assert vfn.flavor
    assert vfn.role == 'attributes'
    # load property information
    from dtk.files import get_file_records
    from drugs.models import Prop
    prop_map={}
    src = get_file_records('prop_dictionary.tsv.master',parse_type='tsv')
    assert next(src) == ['name', 'prop_type', 'multival']
    for rec in src:
        prop_map[rec[0]] = (int(rec[1]),int(rec[2]))
    flag_vals = ('0','1','t','f','true','false')
    # get list of attributes not expected in create files
    # XXX this is somewhat redundant with Collection.static_properties_list;
    # XXX we should provide a single underlying tool somehow
    from drugs.models import Collection,Prop
    # shouldn't include user-modifiable attributes
    bad_attributes = set([
            Prop.OVERRIDE_NAME,
            'synthesis_protection',
            ])
    prefix = Collection.foreign_key_prefix
    for name in prop_map:
        if name.startswith(Collection.foreign_key_prefix):
            # disallow any m_ properties
            bad_attributes.add(name)
            # and disallow the corresponding collection keys
            # (create files should only contain their own collection
            # keys, and those are in the first column, not associated
            # with an attribute name)
            bad_attributes.add(name[len(Collection.foreign_key_prefix):])
        for tox_prefix in ('rt_','pt_'):
            if name.startswith(tox_prefix):
                bad_attributes.add(name)
    # scan file and verify format
    # XXX - support (currently unused) href column?
    last_key=None
    seen_keys=set()
    header=None
    coll_prefix={
            # provide defaults where prefix can't be inferred
            'selleckchem':'SLK',
            }.get(file_class)
    import re
    from collections import defaultdict
    prop_counts = defaultdict(int)
    for rec in get_file_records(path, progress=True):
        if not header:
            header=rec
            expected = [file_class+'_id','attribute','value']
            assert header == expected, \
                    "bad header:"+repr(rec) + " expected: "+repr(expected)
            continue

        if coll_prefix is None:
            # try to infer prefix from leading alphas in first record
            coll_prefix = re.match(r'([A-Z]+).+', rec[0]).group(1)
        else:
            assert rec[0].startswith(coll_prefix) \
                ,"key prefix not '"+coll_prefix+"':"+repr(rec)

        # canonical must be first; all drug records must be consecutive
        if rec[1] == 'canonical':
            assert rec[0] != last_key,"bad key:"+repr(rec)
            assert rec[0] not in seen_keys,"duplicate key:"+repr(rec)
            last_key = rec[0]
            seen_keys.add(rec[0])
            seen_attrs=set()
        else:
            assert rec[0] == last_key,"bad key:"+repr(rec)
        # verify field type rules are followed
        prop_name = rec[1].lower()
        prop_counts[prop_name] += 1
        prop_type,multival = prop_map[prop_name]
        if not multival:
            assert prop_name not in seen_attrs,"repeated attr:"+repr(rec)
        seen_attrs.add(prop_name)
        if prop_type == Prop.prop_types.TAG:
            assert len(rec[2]) <= 256,"bad value:"+repr(rec)
            assert is_utf8(rec[2]),"bad value:"+repr(rec)
        elif prop_type == Prop.prop_types.FLAG:
            assert rec[2].lower() in flag_vals,"bad value:"+repr(rec)
        elif prop_type == Prop.prop_types.INDEX:
            assert is_int(rec[2]),"bad value:"+repr(rec)
        elif prop_type == Prop.prop_types.METRIC:
            assert is_float(rec[2]),"bad value:"+repr(rec)
        elif prop_type == Prop.prop_types.BLOB:
            assert is_utf8(rec[2]),"bad value:"+repr(rec)
        else:
            assert False ,"bad prop name:"+repr(rec)
        if prop_name in bad_attributes:
            assert False ,"disallowed prop name:"+repr(rec)

    # If we have any structural information in the collection, we should have std_smiles.  Some small fraction might be missing due to parsing issues.
    assert prop_counts['smiles_code'] == 0 or prop_counts['std_smiles'] >= prop_counts['smiles_code']*0.9, f'Expected more std_smiles (found {prop_counts["std_smiles"]})'
    print("Prop counts:", dict(prop_counts))
    print(f'{len(seen_keys)} drugs checked')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='verify create file')
    parser.add_argument('filename')
    args = parser.parse_args()

    verify(args.filename)
