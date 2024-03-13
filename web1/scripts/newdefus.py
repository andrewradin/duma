#!/usr/bin/env python3

import sys
import logging
logger = logging.getLogger(__name__)
from path_helper import PathHelper
import os

def single_score(ref_agent, agent_scores, sim):
    return (1.0-agent_scores[ref_agent][0]) * agent_scores[ref_agent][1] * sim**2

def defaultdict_to_dict(d):
    from collections import defaultdict
    if isinstance(d, defaultdict):
        return {k: defaultdict_to_dict(v) for k,v in d.items()}
    else:
        return d

def score_by_similarity(agent_scores, sim_data):
    from collections import defaultdict
    # agent -> simtype -> best_sim_value
    out_scores = defaultdict(lambda: defaultdict(lambda: -1e99))
    out_connections = defaultdict(dict)

    from tqdm import tqdm
    for sim_type, sim_type_data in sim_data.items():
        for ref_agent in tqdm(sim_type_data.row_names, desc=f'{sim_type} scoring', delay=1, mininterval=1):
            for sim_agent, sim_val in sim_type_data[ref_agent].items():
                score = single_score(ref_agent, agent_scores, sim_val)
                if score > out_scores[sim_agent][sim_type]:
                    out_scores[sim_agent][sim_type] = score
                    out_connections[sim_agent][sim_type] = ref_agent

    return defaultdict_to_dict(out_scores), defaultdict_to_dict(out_connections)

def run(input_pickle, settings_pickle, out_file, cores, out_sims):
    import time
    import pickle
    ts = time.time()
    with open(input_pickle, 'rb') as handle:
        in_data = pickle.load(handle)
    with open(settings_pickle, 'rb') as handle:
        settings = pickle.load(handle)
    
    ms = make_metasim(settings)
    output, ref_sims, sims_details = run_data(ms, in_data, precomputed={}, cores=cores)

    import isal.igzip as gzip
    with gzip.open(out_file, 'wb') as f:
        pickle.dump(output, f)

    from dtk.arraystore import put_array
    for method_name, sim_arr in ref_sims.items():
        metadata = {
            'row_names': sim_arr.row_names,
            'col_names': sim_arr.col_names,
        }
        if method_name in sims_details:
            metadata['details'] = sims_details[method_name]
        import numpy as np
        put_array(out_sims, method_name, sim_arr.mat, metadata=metadata, dtype=np.float32, quantize_digits=3)


    print("Took: " + str(time.time() - ts) + ' seconds')

def make_metasim(settings):
    from dtk.metasim import MetaSim
    return MetaSim(
        ppi_choice=settings['ppi'],
        sim_choice=settings.get('sim_db_choice', 'default.v1'),
        thresholds=settings['min_sim'],
        std_gene_list_set=settings['std_gene_list_set'],
        d2ps_method=settings['d2ps_method'],
        d2ps_threshold=settings['d2ps_threshold'],
    )

def run_data(ms, in_data, precomputed, cores):
    ref_similarities, extra_sim = ms.run_all_keys(
        ref_keys=in_data['ref_sim_keys'],
        agent_keys=in_data['ws_sim_keys'],
        methods=in_data['methods'],
        precomputed=precomputed,
        cores=cores,
        )
    
    scores, connections = score_by_similarity(in_data['ref_scores'], ref_similarities)

    out = {
        'scores': scores,
        'connections': connections,
    }

    return out, ref_similarities, extra_sim


if __name__ == "__main__":
    from dtk.log_setup import setupLogging
    setupLogging()

    import argparse

    parser = argparse.ArgumentParser(description='Score drugs with prot_rank')
    parser.add_argument("input_pickle", help="Pickled list of drugs, their p-vals and their enrichments")
    parser.add_argument("settings_pickle", help="Pickled dict of settings")
    parser.add_argument("out_file", help="file to report results")
    parser.add_argument("cores", type=int, help="Number of available cores")
    parser.add_argument("--out-sims", help="file to report similarity data")

    args=parser.parse_args()
    run(**vars(args))
