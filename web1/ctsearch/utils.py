from __future__ import print_function
import six
def get_ct_search_file(version_defaults):
    from dtk.s3_cache import S3File
    file_class='aact'
    s3_file = S3File.get_versioned(
            file_class,
            version_defaults[file_class],
            role='disease_drug_matrix',
            )
    s3_file.fetch()
    return s3_file.path()

class ClinicalTrialsSearch:
    phases=[
            'Phase 0',
            'Phase 1',
            'Phase 1/Phase 2',
            'Phase 2',
            'Phase 2/Phase 3',
            'Phase 3',
            'Phase 4',
            'N/A',
            ]
    def passes_filter(self,rec):
        if rec[self.cond_idx].lower() != self.disease.lower():
            return False
        if rec[self.phase_idx] not in self.phases:
            return False
        if self.after and rec[self.start_idx]:
            if int(rec[self.start_idx]) < self.after:
                return False
        if self.completed and rec[self.status_idx] != 'Completed':
            return False
        return True
    def __init__(
            self,
            disease,
            phases,
            after,
            completed,
            drug,
            version_defaults,
            return_all = False,
            ):
        self.disease = disease
        self.phases = phases
        self.after = after
        self.completed = completed
        from dtk.files import get_file_records
        src = get_file_records(get_ct_search_file(version_defaults))
        header = next(src)
        self.drug_idx = header.index('DRUGS')
        self.study_idx = header.index('STUDY')
        self.treat_idx = header.index('INTERVENTION')
        self.cond_idx = header.index('DISEASE')
        self.phase_idx = header.index('PHASE')
        self.start_idx = header.index('START_YEAR')
        self.status_idx = header.index('OVERALL_STATUS')
        if drug:
            self.get_studies_for_drug(src,drug)
        elif return_all:
            self.get_all_studies(src)
        else:
            self.get_study_ids_by_drug(src)
    def get_studies_for_drug(self,src,drug):
        self._get_studies(src,drug)
    def get_all_studies(self,src):
        self._get_studies(src,None)
    def _get_studies(self,src,drug):
        by_study={}
        from collections import namedtuple
        # NOTE: in the record below:
        # - 'interventions' is a set of all the actual intervention strings
        #   recorded for the study (which may contain multiple drugs and
        #   other descriptive information)
        # - 'drugs' is a set of all extracted drug names, lower-cased;
        #   extraction is done by the ETL
        Study=namedtuple(
                'Study',
                'study phase start status conditions interventions drugs',
                )
        for rec in src:
            if not self.passes_filter(rec):
                continue
            if drug and drug not in rec[self.drug_idx:]:
                continue
            nct_id = rec[self.study_idx]
            t=by_study.setdefault(nct_id,Study(
                            nct_id,
                            rec[self.phase_idx],
                            rec[self.start_idx],
                            rec[self.status_idx],
                            set(),
                            set(),
                            set(),
                            ))
            t.conditions.add(rec[self.cond_idx])
            t.interventions.add(rec[self.treat_idx])
            t.drugs.add(rec[self.drug_idx])
        self.study_list=sorted(
                list(by_study.values()),
                key=lambda x:x.start,
                reverse=True,
                )
    def get_study_ids_by_drug(self,src):
        by_drug={}
        for rec in src:
            if not self.passes_filter(rec):
                continue
            for drug in rec[self.drug_idx:]:
                by_drug.setdefault(drug,set()).add(rec[self.study_idx])
        self.by_drug=list(six.iteritems(by_drug))
        self.by_drug.sort(key=lambda x:len(x[1]),reverse=True)

def format_ct_study_table(study_list):
    from dtk.table import Table
    import dtk.url
    ct_url=dtk.url.clinical_trials_url
    show_set=lambda x:', '.join(x)
    from dtk.html import link
    col_list = [
            Table.Column('Study',
                    cell_fmt=lambda x:link(x,ct_url(x),new_tab=True),
                    ),
            Table.Column('Phase',
                    ),
            Table.Column('Start',
                    ),
            Table.Column('Status',
                    ),
            Table.Column('Conditions',
                    cell_fmt=show_set,
                    ),
            Table.Column('Interventions',
                    cell_fmt=show_set,
                    ),
            ]
    return Table(study_list,col_list)

def ct_study_stats(dn_set):
    from django.db.models import Count
    result = list(
            dn_set.values_list('status').annotate(Count('status'))
            )
    result.sort(key=lambda x:x[0])
    from .models import CtDrugName
    return [
            (CtDrugName.status_vals.get('label',x[0]),x[1])
            for x in result
            ]

