#!/usr/bin/env python

def run(studies, output):
    from dtk.gwas_filter import k_is_good

    from dtk.files import get_file_lines
    from atomicwrites import atomic_write

    out_lines = set()
    filtered_out = set()
    seen = set()
    for study_file in studies:
        for line in get_file_lines(study_file, keep_header=False):
            k = line.split('\t')[0]
            if k in seen:
                continue
            seen.add(k)
            if k_is_good(k):
                out_lines.add(line)
            else:
                filtered_out.add(k)
    with atomic_write(output, overwrite=True) as out:
        for line in sorted(out_lines):
            out.write(line)
    
    print(f"Have {len(out_lines)} good studies")
    print(f"Filtered out {len(filtered_out)} studies with bad keys.  Usually just means missing PMIDs.")
    print('Sample of data filtered out:\n\t' + '\n\t'.join(list(filtered_out)[:10]))

if __name__=='__main__':
    import argparse
    arguments = argparse.ArgumentParser(description="Parse full GRASP DB into our format")
    arguments.add_argument("studies", nargs='*', help="studies data")
    arguments.add_argument("-o", "--output", help="combined studies output file")
    args = arguments.parse_args()
    run(**vars(args))