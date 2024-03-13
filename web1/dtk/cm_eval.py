

def summarize_enrichment(df, baseline_rows, param_cols, score_col, group_col='ws'):
    """Returns new dataframes that have computed deltas from baseline, and summarized by grouping"""
    import pandas as pd
    baseline_df = df[baseline_rows]
    df_out = df.copy()
    delta_col = score_col + ' deltas'
    for group in pd.unique(df[group_col]):
        baseline_val = baseline_df[baseline_df[group_col] == group][score_col]
        group_df = df[df[group_col] == group]
        deltas = group_df[score_col] - baseline_val.values[0]
        df_out.loc[df[group_col] == group, delta_col] = deltas
    
    summarized = df_out[param_cols + [delta_col]].groupby(param_cols)
    mean = summarized.mean()
    std = summarized.std()
    mean[delta_col + ' std'] = std[delta_col]
    mean[delta_col + ' mean'] = mean[delta_col]
    mean.drop(columns=[delta_col], inplace=True)

    return df_out, mean



def add_enrichment_columns(df, metric_name, evalset, codes):
    from dtk.enrichment import MetricProcessor
    processor = MetricProcessor()
    from collections import defaultdict
    scores = defaultdict(list)
    for jid, ws_id in zip(df['jid'], df['ws_id']):
        for code in codes:
            try:
                score = processor.compute(metric=metric_name, ws_or_id=ws_id, job_or_id=jid, code=code, ktset=evalset)
                scores[code].append(score)
            except Exception:
                import traceback as tb
                tb.print_exc()
                scores[code].append(None)
    
    for name, scorelist in scores.items():
        df[name] = scorelist

def load_experiment_settings(fn):
    """
    Settings files list the sets of settings to run combinatorially.
    e.g.:
        ws: [43, 105, 23],
        lambda: [0.01, 0.1, 1.0],
        norm: ["l1", "l2"]
         
        Will return all  3 * 3 * 2 combinations of settings as a list like:
            [
            {ws: 43, lambda: 0.01, norm: "l1",
            {ws: 43, lambda: 0.1, norm: "l1",
            ...
            ]
    """
    def combine(items):
        if not items:
            return [{}]
        
        key, values = items[0]
        assert not isinstance(values, str), "Used a string instead of list"
        after = combine(items[1:])
        out = []
        for value in values:
            for other_vals in after:
                out.append({
                    key: value,
                    **other_vals
                })
        return out

            
    import json
    with open(fn) as f:
        data = json.loads(f.read())
        return combine(list(data.items()))


