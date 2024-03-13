class ChemblIndications:
    def _get_path(self):
        from dtk.s3_cache import S3File, S3Bucket
        # Grab the max chembl version from the path names, stripping off the 'v'
        versions = [int(x.split('.')[2][1:]) for x in S3Bucket('chembl').list()]
        version = max(versions)
        
        s3f=S3File('chembl',f'chembl.full.v{version}.indications.tsv')
        s3f.fetch()
        return s3f.path()
    def get_disease_info(self):
        from dtk.files import get_file_records
        header = None
        items={}
        counts={}
        for rec in get_file_records(self._get_path()):
            if not header:
                header = rec
                mesh_idx = header.index('mesh_id')
                name_idx = header.index('indication')
                continue
            code = rec[mesh_idx]
            items[code] = rec[name_idx]
            if code in counts:
                counts[code] += 1
            else:
                counts[code] = 1
        return items,counts
    def search(self,names=[],codes=[],min_phase=None,include_phase=False):
        from dtk.files import get_file_records
        header = None
        for rec in get_file_records(self._get_path()):
            if not header:
                header = rec
                mesh_idx = header.index('mesh_id')
                disease_idx = header.index('indication')
                drug_idx = header.index('drugname')
                chembl_idx = header.index('chembl_id')
                phase_idx = header.index('max_phase')
                extract = [drug_idx,chembl_idx]
                if include_phase:
                    extract.append(phase_idx)
                continue
            if min_phase is not None and int(rec[phase_idx]) < min_phase:
                continue
            if rec[disease_idx].lower() in names or rec[mesh_idx] in codes:
                yield (rec[x] for x in extract)
