#!/usr/bin/env python
def filter_manifest_urls(infile,outfile):
	#this function writes out one set of files per phenotype description
	#preferentially those that used both_sexes
	header = None
	with open(outfile, 'w') as of:
		for man in get_file_records(infile, parse_type = 'tsv',):
			if not header:
				header = man
				pheno_ind = header.index("Phenotype Code")
				sex_ind = header.index("Sex")
				description_ind = header.index("Phenotype Description")
				last_description = man[description_ind]
				url_ind = header.index("Dropbox File")
				file_ind = header.index("File")
				towrite='\t'.join(["Phenotype Code", "Phenotype Description", "Sex", "File", "Dropbox File"])+'\n'
				continue
			#We also want to ignore the first several lines that are not phenotype information
			if man[pheno_ind] == "N/A":
				continue
			cur_description = man[description_ind]
			#if we've moved on to a new set of phenotype descriptions
			if cur_description != last_description:
				#print out for the last set
				of.write(towrite)
				#and reset things
				last_description = cur_description
				both_flag=0
			#and go through the current line
			#if the phenotype code says "raw", we don't want it
			if "raw" in man[pheno_ind]:
				continue
			else:
				#if we've already found the both_sexes entry, then skip
				if both_flag == 1:
					continue
				else:
					#otherwise, set the towrite line to the current line
					parsed_description = re.sub(' ','_', man[description_ind])
					towrite='\t'.join([man[pheno_ind], parsed_description, man[sex_ind], man[file_ind], man[url_ind]])+'\n'
					#and switch the flag if it's actually the both line
					if man[sex_ind] == "both_sexes":
						both_flag = 1




#This progam takes in the UKBB File manifest release and grabs the appropriate urls

if __name__=='__main__':
	import argparse
	import sys
	import re
	try:
		from dtk.files import get_file_records
	except ImportError:
		sys.path.insert(1, (sys.path[0] or '.')+"/../../web1")
		from dtk.files import get_file_records	
	#=================================================
	# Read in the arguments/define options
	#=================================================
	arguments = argparse.ArgumentParser(description="Parse UKBB File Manifest Release into URLS to download")
	arguments.add_argument("-i", help = "Input Manifest file name")
	arguments.add_argument("-o", help="Output to download file name")
	args = arguments.parse_args()
	# Run the script	
	filter_manifest_urls(args.i,args.o)