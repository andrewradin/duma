
from browse.models import Protein
import re

def find_by_gene(gene_name):
    return Protein.objects.filter(gene=gene_name)

def bulk_find_by_gene(gene_names):
    return Protein.objects.filter(gene__in=gene_names)

def search_geneprot_by_start(pattern, limit):
    """Finds prots whose name or gene name starts with pattern."""
    from django.db.models import Q
    reached_limit = False
    base = Q(uniprot__istartswith=pattern)
    gene = Q(gene__istartswith=pattern)
    q = base | gene
    # .distinct() is required because we are searching by foreign keys, and we
    # could match on multiple of them.  distinct will 'distinct'ify on the
    # protein-specific columns.
    # e.g. uniprot   gene   (alt_name) (alt_uniprot)
    #       P001     G001    prot1
    #       P001     G001               Q001
    #       P001     G001               Z001
    qs = Protein.objects.filter(q).distinct()
    if limit:
        if len(qs) > limit:
            reached_limit = True
        qs = qs[:limit]

    return list(qs), reached_limit

def search_by_any(pattern, limit):
    """Finds prots whose name, or gene name, or alt name contains pattern."""
    from django.db.models import Q
    reached_limit = False
    base = Q(uniprot__icontains=pattern)
    gene = Q(gene__icontains=pattern)
    alt_uni = Q(
            proteinattribute__attr__name='Alt_uniprot',
            proteinattribute__val__icontains=pattern,
            )
    prot_name = Q(
            proteinattribute__attr__name='Protein_Name',
            proteinattribute__val__icontains=pattern,
            )
    alt_prot_name = Q(
            proteinattribute__attr__name='Alt_Protein_Name',
            proteinattribute__val__icontains=pattern,
            )
    q = base | gene | alt_uni | prot_name | alt_prot_name
    # .distinct() is required because we are searching by foreign keys, and we
    # could match on multiple of them.  distinct will 'distinct'ify on the
    # protein-specific columns.
    # e.g. uniprot   gene   (alt_name) (alt_uniprot)
    #       P001     G001    prot1
    #       P001     G001               Q001
    #       P001     G001               Z001
    qs = Protein.objects.filter(q).distinct()
    if limit:
        if len(qs) > limit:
            reached_limit = True
        qs = qs[:limit]

    return list(qs), reached_limit


def parse_global_data(datastr):
    """Parses global data spreadsheet into a structured format.

    Sample input format (one per line):
        "Drug XYZ <tab> Brodo Prot 2(BRD2 or O1.2.3 or Protein Hunk 1...); Brodo Prot 3(... or ...); ..."

    Sample output format:
    [{
        title: 'Drug XYZ',
        targets: [{
            name: 'Brodo Prot 2',
            ids: ['BRD2', 'O1.2.3.', 'Protein Hunk 1', 'EC 1.2.3']
            }, ...
            ]
    }, ...
    ]
    """
    lines = [line.strip() for line in datastr.split('\n')]
    data = [parse_global_data_line(line) for line in lines if line != '']
    filtered_data = filter_global_data_duplicates(data)
    return filtered_data

def filter_global_data_duplicates(data):
    filtered_data = []
    data_by_name = {}
    for drug_entry in data:
        drug_name = drug_entry['title']
        targets = drug_entry['targets']
        
        if drug_name in data_by_name:
            if data_by_name[drug_name]['targets'] == targets:
                # This drug reduced to the same name as another one,
                # and had the same target list.  We'll just drop it.
                continue
            else:
                # It has different targets, we should keep this and use the
                # full name to differentiate.
                drug_entry['title'] = drug_entry['full_name']
                data_by_name[drug_name]['title'] = data_by_name[drug_name]['full_name']

        data_by_name[drug_name] = drug_entry
        filtered_data.append(drug_entry)

    return filtered_data

class DrugNameReducer:
    regex_list = [
            ('period',re.compile(r'(.*)\.$')),
            ('with',re.compile(r'(.*) w$')),
            ('tablets',re.compile(r'(.*) tablets\b')),
            ('tablets',re.compile(r'(.*) tab\b')),
            ('mg',re.compile(r'(.*) [0-9]+ mg\b')),
            ('hcl',re.compile(r'(.*) hcl\b')),
            ('hcl',re.compile(r'(.*) hydrochloride\b')),
            ('hbr',re.compile(r'(.*) hbr\b')),
            ('hbr',re.compile(r'(.*) hydrobromide\b')),
            ('besylate',re.compile(r'(.*) besylate\b')),
            ('tartrate',re.compile(r'(.*) tartrate\b')),
            ('tartrate',re.compile(r'(.*) bitartrate\b')),
            ('sodium',re.compile(r'(.*) sodium\b')),
            ('phosphate',re.compile(r'(.*) phosphate\b')),
            ('calcium',re.compile(r'(.*) calcium\b')),
            ('succinate',re.compile(r'(.*) succinate\b')),
            ('acetate',re.compile(r'(.*) acetate\b')),
            ('fumarate',re.compile(r'(.*) fumarate\b')),
            ('carbonate',re.compile(r'(.*) carbonate\b')),
            ('potassium',re.compile(r'(.*) potassium\b')),
            ('potassium',re.compile(r'(.*) dipotassium\b')),
            ('depot',re.compile(r'(.*) depot\b')),
            ('xl',re.compile(r'(.*) xl\b')),
            ('xr',re.compile(r'(.*) xr\b')),
            ('srt',re.compile(r'(.*) srt\b')),
            ('lar',re.compile(r'(.*) lar\b')),
            ('hfa',re.compile(r'(.*) hfa\b')),
            ('cq',re.compile(r'(.*) cq\b')),
            ('supplement',re.compile(r'(.*) supplement\b')),
            ('intravenous',re.compile(r'(.*) intravenous\b')),
            ]

    @classmethod
    def reduce_drug_name(cls, name):
        for how,regex in cls.regex_list:
            m = regex.match(name)
            if m:
                name = m.group(1)
        return name

def parse_global_data_target(q):
    if '(' not in q:
        return {'name': q, 'ids': []}

    split_idx = q.index('(')
    name = q[:split_idx]

    q = q[split_idx+1:-1]
    parts = q.split(" or ")
    return {
        'name': name.strip(),
        'ids': [x.strip() for x in parts]
        }

def parse_global_data_line(query):
    title = '' 
    if '\t' in query:
        idx = query.index('\t')
        title = query[:idx]
        query = query[idx+1:]

    full_name = title
    title = DrugNameReducer.reduce_drug_name(full_name)
        
    targets = query.split(';')
    parsed_targets = [
            parse_global_data_target(target.strip())
            for target in targets
            ]
    return {
        'title': title,
        'full_name': full_name,
        'targets': parsed_targets
        }

def find_protein_by_name(name):
    return Protein.objects.filter(
            proteinattribute__attr__name__in=(
                    'Protein_Name',
                    'Alt_Protein_Name',
                    ),
            proteinattribute__val=name,
            )

def find_protein_for_global_data_target(t):
    d = parse_global_data_target(t)
    id_list = d['ids'] + [d['name']]
    candidates = {x.uniprot:x for x in bulk_find_by_gene(id_list)}
    for v in id_list:
        l = find_protein_by_name(v)
        if len(l) == 1:
            candidates[l[0].uniprot] = l[0]
    if len(candidates) != 1:
        # XXX we could be more aggressive here
        # XXX - if there are a small number of candidates, or no candidates but
        # XXX   a name match with a small number of values, format a placeholder
        # XXX   as "P54646 or Q13131" 
        # XXX - see if any IDs are gene family name prefixes (HRH for HRH1,2,..
        # XXX   or IL2R for IL2RA,B,...)
        # XXX but just leaving the original string works just as well
        return None
    return next(iter(candidates.values()))
