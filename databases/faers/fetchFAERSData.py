#!/usr/bin/env python3

import sys
# retrieve the path where this program is located, and
# use it to construct the path of the website_root directory
try:
    from path_helper import make_directory
except ImportError:
    sys.path.insert(1, (sys.path[0] or '.')+"/../../web1")
    from path_helper import make_directory

import os

class faers_urls:
    def __init__(self):
        self.dir_name = "urls"
        self.host = 'https://fis.fda.gov/'
        self.directory_url = 'extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html'
        import re
        self.archive_url=re.compile(r'.*_ascii_([0-9]{4}[Qq][1-4])\.zip')
    def run(self):
        # Scrape the latest FDA FAERS page for links to FAERS zip files
        import requests
        content=requests.get(self.host+self.directory_url).content
        from lxml import html
        dom = html.fromstring(content)
        found = 0
        new = 0
        for node in dom.iter("a"):
            href = str(node.get("href"))
            m = self.archive_url.match(href)
            if m:
                found += 1
                fn = m.group(1).upper()
                path = os.path.join(self.dir_name, fn)
                if not os.path.exists(path):
                    with open(path,'w') as fh:
                        fh.write(href+'\n')
                    new += 1
        print('found %d archive links, %d new\n'%(found,new))
        assert found

if __name__=='__main__':
    import argparse

    #=================================================
    # Read in the arguments/define options
    #=================================================

    # get exit codes
    arguments = argparse.ArgumentParser(description="Parse the FAERs HTML")
    args = arguments.parse_args()

    fu = faers_urls()
    fu.run()
