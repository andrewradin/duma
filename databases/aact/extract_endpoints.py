#!/usr/bin/env python

import django_setup

# See description at bottom

def filtered_aact_records(fn,trial_ids):
    header = None
    from dtk.parse_aact import aact_file_records
    for fields in aact_file_records(fn):
        if not header:
            header = [x.upper() for x in fields]
            trial_id_index = header.index('NCT_ID')
            continue
        if fields[trial_id_index] not in trial_ids:
            continue
        yield dict(zip(header,fields))

def build_lookup(fn,trial_ids):
    return { d['ID']:d for d in filtered_aact_records(fn,trial_ids) }

def show_table(fn,columns,trial_ids):
    from dtk.text import print_table
    rows = [
        [d[x] for x in columns]
        for d in filtered_aact_records(fn,trial_ids)
        ]
    print_table([columns]+rows)

def get_trial_ids(source_id):
    from ktsearch.models import KtSearchResult
    qs = KtSearchResult.objects.filter(
            query_id=source_id,
            )
    return set(x.href.split('/')[-1] for x in qs)

def list_trial_info(trial_ids):
    # show interventions along with study id in header line
    from dtk.data import MultiMap
    intervention_lookup = MultiMap(
            (d['NCT_ID'],d['MESH_TERM'])
            for d in filtered_aact_records('browse_interventions.txt',trial_ids)
            ).fwd_map()
    # The database doesn't appear to be fully normalized -- multiple
    # result_group_ids and outcome_ids map to the same outcome or group
    # title. We flatten it here to get more coherent output, but also
    # complain if we find duplicate keys in the flattened mapping.
    #
    # There are also cases where the titles appear to differ by formatting
    # only (e.g. 'Apremilast 30 mg' vs 'Apremilast 30mg' in NCT01307423).
    # I don't attempt to fix those here, but they would be an issue for
    # any automatic processing.
    # XXX - maybe show result group descriptions between NCT_ID and table;
    # XXX   this helps if the titles are things like 'Group 1'
    outcome_lookup = build_lookup('outcomes.txt',trial_ids)
    result_group_lookup = build_lookup('result_groups.txt',trial_ids)
    measure_lookup = {}
    for meas in filtered_aact_records('outcome_measurements.txt',trial_ids):
        t = measure_lookup.setdefault(meas['NCT_ID'],{})
        try:
            rg_title = result_group_lookup[meas['RESULT_GROUP_ID']]['TITLE']
        except KeyError:
            rg_title = ''
        key = (
                rg_title,
                outcome_lookup[meas['OUTCOME_ID']]['TITLE'],
                meas['CLASSIFICATION'],
                meas['CATEGORY'],
                );
        if key in t:
            print('key not unique:',key,
                    'was',t[key]['PARAM_VALUE'],
                    'now',meas['PARAM_VALUE'],
                    )
            #print('  ',t[key])
            #print('  ',meas)
        #assert key not in t
        t[key] = meas
    from dtk.text import print_table,wrap,ljustify,split_multi_lines
    for trial_id,data in measure_lookup.items():
        print()
        interventions = sorted(intervention_lookup.get(trial_id,[]))
        print(trial_id,interventions)
        groups = set(x[0] for x in data.keys())
        groups = sorted(groups)
        header = ['measure']+groups
        measures = set(x[1:] for x in data.keys())
        measures = sorted(measures)
        rows = [header]
        for measure in measures:
            label = ' '.join(x for x in measure if x)
            row = [label]
            for group in groups:
                meas = data.get((group,)+measure)
                row.append(meas['PARAM_VALUE'] if meas else '')
            rows.append(row)
        if True:
            wrap_col = 0
            # at least some labels contain unicode chars that wrap to multiple
            # columns (e.g. &#65289 FULLWIDTH_RIGHT_PARENTHESIS in NCT01680159).
            # convert any non-ascii chars here to avoid wrapping alignment
            # issues
            for row in rows:
                row[wrap_col] = ''.join(
                        x if x.isascii() else '_' for x in row[wrap_col]
                        )
            wrap(rows,wrap_col,50)
            rows = split_multi_lines(rows)
            ljustify(rows,wrap_col)
        print_table(rows)

def list_searches(ws_id):
    from ktsearch.models import KtSource
    qs = KtSource.objects.filter(
            source_type='AACTKtSource',
            search__ws_id=ws_id,
            )
    from dtk.text import print_table,fmt_time
    import json
    print_table([('source id','term','date','result count')]+[
            (
                str(src.id),
                json.loads(src.config)['search_term'],
                fmt_time(src.search.created),
                str(src.ktsearchresult_set.count()),
            )
            for src in qs
            ])

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description='''\
Extract endpoint data for examination.

This is an attempt to assess the amount of work required to parse endpoint
data (e.g. to determine relative efficacy or toxicity).

It exposes several issues with AACT data:
- the database isn't fully normalized, and labels are not necessarily
  consistent, so particular measures and study arms don't always group
  together
- there are often large numbers of endpoint measurements, so determining the
  most relevant ones can be difficult, and may require manual intervention
- data is often longitudinal; this requires heuristic string parsing to
  order it correctly
- similarly, study arms often differ by dose, and require parsing to order
  correctly
- some results are reported as patient counts with some response; these would
  be easier to interpret if converted to a percentage of patients
- arm labels may be abstract ('Group 1' instead of the treatment name); more
  special case parsing code would be required to make these more interpretable
- arm labels and measure labels are not usually consistent enough across
  studies to allow for easy aggregation
''',
            )
    parser.add_argument('--ws-id')
    parser.add_argument('--source-id')
    parser.add_argument('--trial-id')
    args = parser.parse_args()
    if args.ws_id:
        list_searches(args.ws_id)
    if args.source_id:
        trial_ids = get_trial_ids(args.source_id)
        list_trial_info(trial_ids)
    if args.trial_id:
        list_trial_info(set([args.trial_id]))

