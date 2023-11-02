#!/usr/bin/env python

import sys
single_vals = [
            'UniProtKB-ID'
           ,'Gene_Name'
           ,'UniRef90'
           ]
list_vals = [
	        'Gene_Synonym'
           ,'STRING'
           ,'GeneCards'
           ,'KEGG'
           ,'Reactome'
           ,'MIM'
           ,'GeneID'
           ,'Ensembl'
           ,'Ensembl_TRS'
           ,'Ensembl_PRO'
           ]

def get_best_uniprot(all_set, data_dict):
    id = None
    max = 0
    for x in all_set:
        if data_dict[x]['id_line_count'] > max or (data_dict[x]['id_line_count'] == max and ('Gene_Name' in data_dict[x].keys() and 'Gene_Name' not in data_dict[id].keys())):
            id = x
            max = data_dict[x]['id_line_count']
    if not id:
        sys.stderr.write("Unable to find anything for these: " + " ".join(list(all_set)) + "\n")
    all_set.remove(id)
    return id,all_set

all_data = {}
uniref90 = {}
# this is basically the inverse of uniref90
not_ur90_keys = set()
id_line_count = 0
current_id = None
print "\t".join(['Uniprot', 'attribute', 'value'])
for line in sys.stdin:
    fields = line.rstrip("\n").split("\t")
    id_line_count += 1
    if fields[1] == 'UniProtKB-ID':
        ### for ranking clusters without a human UniRef90 ID we count the
        ### number of lines of data as a proxy for how well an entry is researched
        if current_id:
            all_data[current_id]['id_line_count'] = id_line_count
            if current_id != uniref90_key:
                not_ur90_keys.add(current_id)
            try:
                uniref90[uniref90_key].add(current_id)
            except KeyError:
                uniref90[uniref90_key] = set([current_id])
        current_id = fields[0].split("-")[0] # remove isoform info
        id_line_count = 0
        all_data[current_id] = {}
        all_data[current_id][fields[1]] = fields[2][:-6] # strip _HUMAN
        ### not all entries have a UniRef90 cluster, this resulted in these proteins
        ### being excluded from the final list.
        ### To avoid this, we will set a default value for the UniRef90:
        ### The Uniprot ID of the protein. Then, if the entry does have a UniRef90,
        ### that will be overwritten, but if not, we'll still get the entry.
        uniref90_key = current_id
    elif fields[1] in list_vals:
        try:
            all_data[current_id][fields[1]].add(str(fields[2]))
        except KeyError:
            all_data[current_id][fields[1]] = set([str(fields[2])])
    elif fields[1] == 'UniRef90':
        uniref90_key = fields[2][9:].split("-")[0] # strip UniRef90_ and remove isoform info
    elif fields[1] in single_vals:
        all_data[current_id][fields[1]] = fields[2]

# take care of the last one
all_data[current_id]['id_line_count'] = id_line_count
if current_id != uniref90_key:
    not_ur90_keys.add(current_id)
try:
    uniref90[uniref90_key].add(current_id)
except KeyError:
    uniref90[uniref90_key] = set([current_id])

### For ~100 proteins we were running into a situation where the ID (e.g. protX)
### we just ID'd was a Uniprot that was a Uniref90 key.
### Normally this isn't possible (b/c protX didn't even map to itself, but a non-human ID)
### The only case I've seen happen is when an isoform of protX is the key
### but we strip that info away and then get issues.
### To address this we trace which uniprot IDs mapped to someother Uniref90 key
### those uniprots are then no longer considered a reasonable id
### Basically we found something wonky with the UniRef90 clustering,
### and took the conservative approach of separating the questionable parts of the cluster
mistakes = [x for x in not_ur90_keys if x in uniref90]
sys.stderr.write(" ".join(["Found", str(len(mistakes)), "unexpected keys. Attempting to fix those.\n"]))
for x in mistakes:
    id,alts = get_best_uniprot(uniref90[x], all_data)
    uniref90[id] = alts
    del uniref90[x]

for id, alts in uniref90.items():
    try:
        assert all_data[id]['UniProtKB-ID']
        alts.discard(id)
    except (KeyError,AssertionError):
        ### These are weird proteins that cluster with non-Human Uniprots.
        ### When there are several of them we can try to condense ourselves
        id,alts = get_best_uniprot(alts, all_data)
    print "\t".join([id, 'UniProtKB-ID', all_data[id]['UniProtKB-ID']])
    try:
        name = all_data[id]['Gene_Name']
        print "\t".join([id, 'Gene_Name', all_data[id]['Gene_Name']])
    except KeyError:
        name = None
    for alt in alts:
        print "\t".join([id, 'Alt_uniprot', alt])
        if name and 'Gene_Name' in all_data[alt].keys() and all_data[alt]['Gene_Name'] != name:
            try:
                all_data[id]['Gene_Synonym'].add(all_data[alt]['Gene_Name'])
            except KeyError:
                all_data[id]['Gene_Synonym'] = set([all_data[alt]['Gene_Name']])
    for k in list_vals:
        gen = (a for a in alts if k in all_data[a])
        for alt in gen:
            if k == 'Gene_Synonym':
                gen2 = (gs for gs in all_data[alt][k] if gs != name)
                for gs in gen2:
                    try:
                        all_data[id][k].add(gs)
                    except KeyError:
                        all_data[id][k] = set([gs])
            elif k in all_data[id]:
                all_data[id][k].update(all_data[alt][k])
            else:
                all_data[id][k] = all_data[alt][k]
        try:
            print "\n".join(["\t".join([id, k, x]) for x in all_data[id][k]])
        except KeyError:
            pass
