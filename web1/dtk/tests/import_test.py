

def test_imports():
    # For the files that we don't have proper tests for, at least check that we can import them without error.
    from dtk import cas, cm_eval, coll_condense, combo, ext_label, mol_struct, reactome, refresh_group, scaling, sms, umls

    #from ML import run_eval_sklearn

    import feature_set2
    import dea_simple

    from runner import drive_background, plot, run_process

    from scripts import build_jobfile, build_xws_parm_eval_script, duma_version_audit, lts_clean, multi_compare, multi_refresh