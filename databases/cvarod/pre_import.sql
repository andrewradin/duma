-- delete any existing database
drop database if exists canAdverseReportsData;
create database canAdverseReportsData;
use canAdverseReportsData;

-- in that database, make a new table called drug that has the number of events, and then the name of the drug
create table drugsFromCVAROD (
	event int,
	drug varchar(80)
);
 
-- Create a second table that also has a number occurrences, but this time for indications (i.e. diseases)
create table diseasesFromCVAROD (
	event int,
	indi varchar(80)
);
 
-- Create a third table that also has a number occurrences, but this time for demographics (i.e. demographic)
create table demoFromCVAROD (
	event int,
	age_yr float null,
	weight_kg float null,
	sex varchar(1) null
);
