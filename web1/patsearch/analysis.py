


class ParsedContent(object):
    def __init__(self, content):
        abstracts = content.get('abstract_localized', [])
        titles = content.get('title_localized', [])
        claims = content.get('claims_localized', [])
        
        self.abstract = self._pick(abstracts)
        self.title = self._pick(titles)
        self.claims = self._pick(claims)

    def _pick(self, entries):
        for entry in entries:
            if entry['language'] == 'en':
                return entry['text']

        if len(entries) > 0:
            return entries[0]['text']

        return ''

def find_all(needle, hay):
    import re
    for m in re.finditer(r'\b'+re.escape(needle)+r'\b', hay, re.IGNORECASE):
        yield m.start(0)
    return

def pull_context(idx, src, term):
    # Maximum context to pull in on either side, in characters.
    MAX_HALF_CONTEXT = 100
    left = src.rfind('.', 0, idx) + 1
    right = src.find('.', idx)
    if right == -1:
        right = len(src)

    left = max(left, idx - MAX_HALF_CONTEXT)
    right = min(right, idx + len(term) + MAX_HALF_CONTEXT)

    term_end = idx + len(term)

    return src[left:idx] + "<b>" + src[idx:term_end] + "</b>" + src[term_end:right]


def analyze_patent(content, disease_terms, drug_terms):
    drug_ev = []
    disease_ev = []
    parsed = ParsedContent(content)
    srcs = [parsed.title, parsed.abstract, parsed.claims]
    for src in srcs:
        for terms, ev in ((drug_terms, drug_ev), (disease_terms, disease_ev)):
            for term in terms:
                idxs = find_all(term, src)
                contexts = [pull_context(idx, src, term) for idx in idxs]
                ev.extend(contexts)
    n_drug = len(drug_ev)
    n_dis = len(disease_ev)
    # Score well for having lots of evidence, particularly if both drug & dis.
    # Emphasize results with both drug & disease terms.
    score = (n_drug/100.0 + n_dis/100.0 + n_drug * n_dis) ** 0.5

    if len(parsed.claims) == 0 and (n_drug == 0 or n_dis == 0):
        # Let's be safe and mark these as unknown score, because we don't
        # have the claims data.
        from patsearch.models import PatentSearchResult 
        score = PatentSearchResult.SCORE_UNKNOWN
    return {
        'score': score,
        'evidence': {
            'drug_ev': drug_ev,
            'disease_ev': disease_ev,
        },
    }
