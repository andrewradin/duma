-- delete any existing database
drop database if exists goTerms;
create database goTerms;
use goTerms;

-- in that database, make a new table called uniprotToGOTerms that has the GO term associated with an UNIPROT ID as well as the type and evidence codes
create table uniprotToGOTerms (
	entrezID varchar(15),
	UniprotID varchar(20),
	GeneOntology varchar(20),
	EvidenceCode varchar(4),
	GOType varchar(3)
);

-- INDEX columns are exactly that. They're indexed to make it faster to look up information using that column
alter table uniprotToGOTerms add index (UniprotID);
