#!/usr/bin/env python
import sys, os
import re

class parse(object):
    def __init__(self, **kwargs):
        self.base_url='https://ncats.nih.gov'
        self.html_url = '/ntu/assets/'
        self.year = kwargs.get('year', None)
        self.conv_file = kwargs.get('cf', None)        
        try:
            from path_helper import make_directory
        except ImportError:
            sys.path.insert(1, (sys.path[0] or '.')+"/../../web1")
            from path_helper import make_directory
        make_directory(self.year)
        self.out_file = '.'.join([self.year, 'collection', 'tsv'])
        self.dpi_file = '.'.join([self.year, 'dpi', 'tsv'])
    def run(self):
        from dtk.url import get_three_part_html
        self.full_html = get_three_part_html(self.base_url, self.html_url, self.year)
        self._get_hrefs()
        self._parse_pdfs()
    def _parse_pdfs(self):
        from dtk.files import get_file_lines
        self._load_converter()
        gene_url_base = '://www.ncbi.nlm.nih.gov/gene/'
        url_begin = 'http'
        gene_url = url_begin + gene_url_base
        gene_url2 = url_begin + 's' + gene_url_base
        self.results = {}
        for href,name in self.pdf_hrefs.iteritems():
            self.results[href] = set()
            txt = self._pdf_to_txt(name, self.base_url+href)
            for l in get_file_lines(txt, grep=[gene_url_base]):
                gen = (w for x in l.split() for w in x.split(';') if w.startswith(url_begin))
                for w in gen:
                    gene_id = re.sub('[^0-9]','', w.lstrip(gene_url).lstrip(gene_url2))
                    # There was one error in the data provided by NCATS
                    # a digit was dropped. Just to keep everything together,
                    # I've fixed that here.
                    # Luckily 41511 isn't a human gene ID,
                    # so the chances of this messing other things up is minimal
                    if gene_id == '41511':
                        gene_id = '415116'
                    if gene_id not in self.gene2uni:
                        print "Unable to find:",gene_id
                        continue
                    unis = self.gene2uni[gene_id]
                    self.results[href].update(unis)
    def _load_converter(self):
        try:
            from parse_disgenet import make_entrez_2_uniprot_map
        except ImportError:
            sys.path.insert(1, (sys.path[0] or '.')+"/../disgenet")
            from parse_disgenet import make_entrez_2_uniprot_map
        self.gene2uni = make_entrez_2_uniprot_map(self.conv_file)
    def _pdf_to_txt(self, name, url):
        import subprocess
        file_base = os.path.join(self.year, name)
        fn = file_base+'.pdf'
        txt = file_base+'.txt'
        if not os.path.isfile(fn):
            f = open(fn, 'w')
            subprocess.call(['curl', url], stdout=f)
            f.close()
        subprocess.check_call(['pdftotext', fn, txt])
        return txt
    def _get_hrefs(self):
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(self.full_html, 'html.parser')
        urls_suffixes = {}
        for link in soup.find_all('a'):
            x = link.get('href')
            if not x:
                continue
            if x.endswith('.pdf'):
                urls_suffixes[x] = x.split('/')[-1].rstrip('.pdf')
        self.pdf_hrefs = urls_suffixes
        print 'got',len(self.pdf_hrefs),'pdf hrefs'
        self.syns = {}
        for link in soup.find_all('p'):
            l = link.contents
            if not l or not re.match(r'\s+', unicode(l[-1])):
                continue
            if unicode(l[0]).startswith('<strong>'):
                link2 = link.a
                if not link2:
                    continue
                k = link2.get('href')
            elif (unicode(l[0]).startswith('<a')):
                k = l[0].get('href')
            else:
                continue
            v = l[-1].strip().split(', ')
            v = [x.lstrip('(').rstrip(')') for x in v]
            if v[0].startswith('PDF -'):
                continue
            self.syns[k] = v
    def _add_key_hyphen(self, href):
        import re
        l = self.syns.get(href, [])
        # split on letter to number transitions
        k_parts = [s for s
                   in re.split('(\D+)',
                               self.pdf_hrefs[href]
                              )
                   if s
                  ]
        if unicode(k_parts[0]).isalpha():
            if k_parts[-1] == '2016' and k_parts[-2].endswith('-'):
                del k_parts[-1]
            l.append("-".join([k_parts[0],
                              "".join(k_parts[1:])
                             ]).rstrip('-')
                    )
        return l
    def dump(self, dpi_evid_default='0.9', dpi_dir_default='0'):
        with open(self.out_file, 'w') as f:
            with open(self.dpi_file, 'w') as f2:
                f.write("\t".join(['ncats_id', 'attribute', 'value']) + "\n")
                f2.write("\t".join(['ncats_id', 'uniprot_id', 'evidence', 'direction']) + "\n")
                for href, k in self.pdf_hrefs.iteritems():
                    f.write("\t".join([k, 'canonical', self.pdf_hrefs[href]]) + "\n")
                    syns = self._add_key_hyphen(href)
                    for syn in syns:
                        f.write("\t".join([k, 'synonym', syn]) + "\n")
                    for uni in self.results.get(href, []):
                        f2.write("\t".join([k, uni, dpi_evid_default, dpi_dir_default]) + "\n")

if __name__ == '__main__':
    import argparse

    arguments = argparse.ArgumentParser(description="Parses NCATS HTML and link PDFs")
    arguments.add_argument("year", help="2012, 2014, or 2017")
    arguments.add_argument('conv_file', help="HUMAN_9606_Uniprot_data.tsv")
    args = arguments.parse_args()

    p = parse(year = args.year, cf = args.conv_file)
    p.run()
    p.dump()
