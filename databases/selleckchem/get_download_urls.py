#!/usr/bin/env python3

from dtk.lazy_loader import LazyLoader
import sys


"""
We seem to run into some cloudflare ddos protection during this scraping.
For now we've added some progressive sleeps & retries to avoid it.
But we should probably just use the pubchem version of this data instead and
make sure we're using it in the same way.
"""

class SelleckLibraryURLs(LazyLoader):
    base_url='https://www.selleckchem.com'
    # Blacklist the ones we know have no xlsx files, to avoid retrying a lot very slowly.
    blacklist=set([
            '/screening/express-pick-library-premium-version.html',
            ])
    def _lib_page_urls_loader(self):
        found_any = False
        while not found_any:
            import requests
            rsp = requests.get(self.base_url+'/screening-libraries.html')
            from lxml import html
            dom = html.fromstring(rsp.content)
            result = set()
            for node in dom.iter('a'):
                href = node.get("href")
                if href.startswith('/screening/'):
                    result.add(href)
                    found_any = True
            if not found_any:
                sys.stderr.write("Didn't find toplevel screeners, sleeping\n")
                import time
                time.sleep(15)
        return result - self.blacklist
    def _xls_urls_loader(self):
        import requests
        from lxml import html
        result = set()
        import time
        from tqdm import tqdm
        for page_url in tqdm(self.lib_page_urls):
            # This is hand-tuned to the rough amount of time cloudflare seems to require us to wait before
            # we can avoid the interstitial, though we will subsequently retry and sleep longer.
            sleep_len = 15
            found_any = False
            while not found_any:
                rsp = requests.get(self.base_url+page_url)
                dom = html.fromstring(rsp.content)
                for node in dom.iter('a'):
                    href = node.get("href")
                    if href.endswith('.xlsx'):
                        result.add(href)
                        found_any = True
                if not found_any:
                    if 'Your browser will redirect to your requested content shortly' not in rsp.content.decode('utf8'):
                        sys.stderr.write(f"Giving up on {page_url}, no xls that we can find")
                        break

                    sys.stderr.write(f"Didn't find any .xlsx links on {self.base_url+page_url}, going to retry\n")
                    if sleep_len > 60:
                        raise Exception(f"Giving up on {page_url}, either it has no xls or we're getting blocked; add to blacklist if it has no xls")
                    time.sleep(sleep_len)
                    sleep_len *= 1.5
                    continue
        return result

if __name__ == "__main__":
    import argparse
    arguments = argparse.ArgumentParser(description="Gets Selleck screening library download URLs")
    
    args = arguments.parse_args()

    slu = SelleckLibraryURLs()
    #print(sorted(slu.lib_page_urls))
    #print(len(slu.lib_page_urls))
    #print(len(slu.xls_urls))
    #print(sorted(slu.xls_urls))
    prefix = 'https://file.selleckchem.com/downloads/library/'
    prefix_len = len(prefix)
    for url in sorted(slu.xls_urls):
        if url.startswith(prefix):
            # output entry for versions.py file list
            print(f"'{url[prefix_len:]}',")
        else:
            # output warning
            print(f"# SKIPPING {url}")

