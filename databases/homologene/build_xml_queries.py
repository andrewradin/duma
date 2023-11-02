#!/usr/bin/env python

### This was a hacky way to recreate the XML suggested at the URL below, while changing the genome used
### Note this just uses the latest genome build for all species
### https://m.ensembl.org/biomart/martview

if __name__=='__main__':

    name_2_file={
                 'mmusculus_gene_ensembl' : 'mouse_query.xml',
                 'rnorvegicus_gene_ensembl' : 'rat_query.xml',
                 'clfamiliaris_gene_ensembl' : 'dog_query.xml',
                 'drerio_gene_ensembl' : 'zebrafish_query.xml',
                }
    from atomicwrites import atomic_write
    for n,fn in name_2_file.items():
        with atomic_write(fn) as f:
            f.write("\n".join([
                              'query=<?xml version="1.0" encoding="UTF-8"?>',
                              '<!DOCTYPE Query>',
                              '<Query  virtualSchemaName = "default" formatter = "TSV" header = "0" uniqueRows = "0" count = "" datasetConfigVersion = "0.6" >',
                              f'\t<Dataset name = "{n}" interface = "default" >',
                              '\t\t<Filter name = "with_entrezgene" excluded = "0"/>',
                              '\t\t<Attribute name = "ensembl_gene_id" />',
                              '\t\t<Attribute name = "ensembl_transcript_id" />',
                              '\t\t<Attribute name = "entrezgene_id" />',
                              '\t</Dataset>',
                              '</Query>'
                              ])
                   )


