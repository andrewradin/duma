#!/usr/bin/env python3

import xml.etree.ElementTree as etree

ns='{http://www.drugbank.ca}'

def handle_drug(drug,result):
    drug_id = "unknown"
    # filter for the drugs we extract
    for drugbankid in drug.findall(ns+'drugbank-id'):
         if drugbankid.get('primary') == 'true':
             drug_id = drugbankid.text
             break
    if drug_id == "unknown":
        return
    result['primary'] += 1
    # count individual fields
    for field in (
            'indication',
            'pharmacodynamics',
            'mechanism-of-action',
            'toxicity',
            'metabolism',
            'absorption',
            'half-life',
            'protein-binding',
            'route-of-elimination',
            'volume-of-distribution',
            'clearance',
            ):
        item = drug.find(ns+field)
        if item.text:
            result[field] += 1
    # count categories
    categories = drug.find(ns+'categories')
    any_mesh = False
    for item in categories.findall(ns+'category'):
        mesh_id = item.find(ns+'mesh-id')
        if mesh_id.text:
            result['mesh-category'] += 1
            any_mesh = True
        else:
            result['other-category'] += 1
    if any_mesh:
        result['with-mesh-category'] += 1
    # count dosage
    dosages = drug.find(ns+'dosages')
    for item in dosages.findall(ns+'dosage'):
        result['with-dosage'] += 1
        break
    # count properties
    for prop_type in ('experimental','calculated'):
        has_any = False
        container = drug.find(ns+prop_type+'-properties')
        if not container:
            continue
        for prop in container.findall(ns+'property'):
            kind = prop.find(ns+'kind')
            result[prop_type+' '+kind.text] += 1
            has_any = True
        if has_any:
            result['with-'+prop_type+'-properties'] += 1

def parse_xml(src):
    from collections import Counter
    result = Counter()
    for event,elem in etree.iterparse(src,events=('start','end')):
        if event == 'end' and elem.tag == ns+'drug':
            result['drug'] += 1
            handle_drug(elem,result)
            elem.clear()
    return result

def open_input(fn):
    if fn.endswith('.gz'):
        if False:
            import gzip
            return gzip.open(fn, 'r')
        else:
            from dtk.files import open_pipeline
            return open_pipeline([['gunzip','-c',fn]])
    return open(fn,'r')

def format_result(result):
    prop_counts = {}
    drug_counts = {}
    print(result['primary'],'drugs examined')
    for k,v in result.most_common():
        if k in ('drug','primary','mesh-category','other-category'):
            pass
        elif ' ' in k:
            # it's a property
            parts = k.split()
            prop_type = parts[0]
            prop_name = ' '.join(parts[1:])
            try:
                d = prop_counts[prop_name]
            except KeyError:
                d = {'calculated':0,'experimental':0}
                prop_counts[prop_name] = d
            d[prop_type] = v
        else:
            label = k[5:] if k.startswith('with-') else k
            if label == 'mesh-category':
                avg = result[label]/v
                label += f' (avg. {avg:.2} per drug)'
            print('  ',v,'with',label)
    print()
    print('Property breakdown:')
    from dtk.text import print_table
    print_table([['property','calculated','experimental']]+[
            [
                    k,
                    str(prop_counts[k]['calculated']),
                    str(prop_counts[k]['experimental']),
                    ]
            for k in sorted(prop_counts)
            ])

################################################################################
#
# program starts here
#
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('xml_file')
    args = parser.parse_args()

    result = parse_xml(open_input(args.xml_file))
    format_result(result)
