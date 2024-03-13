#!/usr/bin/env python3

import django_setup


"""
This script audits changes in our external data sources to molecules that
we have manually modified and added to our DUMA collection.

The DUMA collection is unique in that we ignore all our sources of DPI for
that molecule, which allows us to remove DPI.

However, that means we have to manually incorporate any new data updates.
"""

def pull_dpis(fn, key):
    from dtk.files import get_file_records
    return get_file_records(fn, select=([key], 0), keep_header=False)


def get_dpi_sig(drug, version):
    from dtk.s3_cache import S3Bucket, S3File
    from dtk.files import VersionedFileName
    mol_matches = drug.get_molecule_matches(version)
    match_keys = mol_matches.drug_keys
    if len(match_keys) == 1:
        # This is a duma-only drug.
        #return []
        pass

    match_inputs = mol_matches.match_inputs

    all_dpis = []

    for col, colkey in match_keys:
        assert col.endswith('_id')
        col = col[:-3]

        ip_flavor,ip_version = match_inputs[col].split('.')
        assert ip_version.startswith('v')
        ip_version = int(ip_version[1:])
        meta = VersionedFileName.meta_lookup[col]
        if 'evidence' not in meta.roles:
            continue
        s3b = S3Bucket(col)
        result = []
        for fn in s3b.list(cache_ok=True):
            vfn = VersionedFileName(meta=meta,name=fn)
            if vfn.version != ip_version:
                continue
            if vfn.role != 'evidence':
                continue

            s3f = S3File(s3b, fn)
            s3f.fetch()
            dpis = pull_dpis(s3f.path(), colkey)
            all_dpis.extend(dpis)
    return all_dpis

def generate_diff(sig1, sig2, prot2gene):
    # For each protein, output a row containing all known values &
    # their source
    from collections import defaultdict
    prot2value2sources1 = defaultdict(lambda: defaultdict(set))
    # entries are dpikey(CHEMBLXXX), prot, evid, direction
    for entry in sig1:
        prot2value2sources1[entry[1]][(entry[2],entry[3])].add(entry[0])

    prot2value2sources2 = defaultdict(lambda: defaultdict(set))
    for entry in sig2:
        prot2value2sources2[entry[1]][(entry[2],entry[3])].add(entry[0])


    all_prots = prot2value2sources1.keys() | prot2value2sources2.keys()
    out = []
    for prot in all_prots:
        value2sources1 = prot2value2sources1[prot]
        value2sources2 = prot2value2sources2[prot]

        all_values = value2sources1.keys() | value2sources2.keys()

        for value in all_values:
            sources1 = value2sources1[value]
            sources2 = value2sources2[value]

            def tostr(x):
                if not x:
                    return '<None>'
                else:
                    return ','.join(x)

            unchanged = False

            if sources1 == sources2:
                # No change in evidence for this prot.
                unchanged = True

            has_duma1 = any('DUMA' in x for x in sources1)
            has_duma2 = any('DUMA' in x for x in sources2)
            if sources1 and len(sources1 - sources2) == 0 and has_duma1:
                # We added more sources that agree with this prot,
                # but DUMA already agreed with them before.
                unchanged = True

            if not sources1 and has_duma2:
                # This is something we manually added since the first
                # version, doesn't count as a change to flag.
                unchanged = True

            if not has_duma1 and not has_duma2 and len(sources2 - sources1) == 0:
                # We already removed it, and we didn't add any new evidence
                # to make us change that decision.
                unchanged = True

            evid, direction = value

            gene = prot2gene.get(prot, f'({prot})')
            entry = [prot, gene, evid, direction, tostr(sources1), tostr(sources2), not unchanged]
            out.append(entry)

    out.sort(key=lambda x: (x[1], x[2], x[3])) # Sort by gene, evid, dir
    return out


def diff_drug(drug, version1, version2, prot2gene):
    # For each match, get version1 DPI & version2 DPI.
    sig1 = get_dpi_sig(drug, version1)
    sig2 = get_dpi_sig(drug, version2)
    if sorted(sig1) != sorted(sig2):
        return generate_diff(sig1, sig2, prot2gene)
    else:
        return []

