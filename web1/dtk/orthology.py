

def get_ortho_records(ws, uniprots):
    from dtk.files import get_file_records
    from browse.default_settings import orthologs
    s3_file = orthologs.get_s3_file(ws)
    header = None
    out = []
    for rec in get_file_records(s3_file.path(), keep_header=True, select=(uniprots, 'uniprot')):
        if header is None:
            header = list(rec)
        else:
            out.append(dict(zip(header, rec)))
    return out