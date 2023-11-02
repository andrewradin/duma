from django.db import models
from tools import Enum


# PatentSearch
#   {n}DrugDiseasePatentSearch
#      {1}GooglePatentSearch
#         {n}GooglePatentSearchResult
#             {1}Patent
#      {n}PatentSearchResult

class PatentSearch(models.Model):
    ws = models.ForeignKey("browse.Workspace", null=True, on_delete=models.CASCADE)
    job = models.ForeignKey("runner.Process", null=True, on_delete=models.CASCADE)
    created = models.DateTimeField(auto_now_add=True)
    user = models.CharField(max_length=70)
    query = models.TextField(null=True, blank=True)

    @property
    def drug_disease_patent_searchs(self):
        return self.drugdiseasepatentsearch_set.all()

    @property
    def num_patents(self):
        out = 0
        for drug_search in self.drug_disease_patent_searchs:
            out += len(drug_search.patentsearchresult_set.all())
        return out

    @property
    def num_unresolved(self):
        out = 0
        for drug_search in self.drug_disease_patent_searchs:
            out += len(drug_search.patentsearchresult_set.filter(
                    resolution=PatentSearchResult.resolution_vals.UNRESOLVED
                    ))
        return out


class DrugDiseasePatentSearch(models.Model):
    patent_search = models.ForeignKey(PatentSearch, on_delete=models.CASCADE)
    query = models.TextField()

    # Something to identify this search to the viewer.
    drug_name = models.CharField(max_length=255)

    # Include an associated wsa if we have one.
    wsa = models.ForeignKey("browse.WsAnnotation", null=True, on_delete=models.CASCADE)

    @property
    def patent_search_results(self):
        return self.patentsearchresult_set.all().order_by('-score', 'patent__title')
    
    @property
    def resolution_summary(self):
        from collections import defaultdict
        out = defaultdict(int)
        for result in self.patent_search_results:
            result_status = result.resolution_text
            out[result_status] += 1
        return out

    @property
    def query_data(self):
        import json
        return json.loads(self.query)

class PatentFamily(models.Model):
    """All patents in the same family are the same underlying patent."""
    family_id = models.CharField(max_length=32, primary_key=True)

class Patent(models.Model):
    pub_id = models.CharField(max_length=255, primary_key=True)
    title = models.TextField(null=True, blank=True)
    abstract_snippet = models.TextField(null=True, blank=True)
    date = models.DateField(null=True, blank=True)
    href = models.TextField(null=True, blank=True)

    # We don't know the family ID until we look it up in bigquery.
    family = models.ForeignKey(PatentFamily, blank=True, null=True, on_delete=models.CASCADE)

class GooglePatentSearch(models.Model):
    query = models.TextField()
    total_results = models.IntegerField()
    href = models.TextField()


class GooglePatentSearchResult(models.Model):
    search = models.ForeignKey(GooglePatentSearch, on_delete=models.CASCADE)
    search_snippet = models.TextField()
    patent = models.ForeignKey(Patent, on_delete=models.CASCADE)

class BigQueryPatentSearchResult(models.Model):
    patent = models.ForeignKey(Patent, on_delete=models.CASCADE)

class PatentContentInfo(models.Model):
    # Indicates where the content is stored.
    ws = models.ForeignKey("browse.Workspace", null=True, on_delete=models.CASCADE)
    job = models.ForeignKey("runner.Process", null=True, on_delete=models.CASCADE)

    patent_family = models.ForeignKey(PatentFamily, null=True, on_delete=models.CASCADE)
    has_abstract = models.BooleanField()
    has_claims = models.BooleanField()

class PatentSearchResult(models.Model):
    resolution_vals = Enum([], [
            ('UNRESOLVED',),
            ('RELEVANT',),
            ('IRRELEVANT_DRUG',),
            ('IRRELEVANT_DISEASE',),
            ('IRRELEVANT_ALL',),
            ('NEEDS_MORE_REVIEW',),
            ('SKIPPED',),
            ])
    search = models.ForeignKey(DrugDiseasePatentSearch, on_delete=models.CASCADE)
    patent = models.ForeignKey(Patent, on_delete=models.CASCADE)
    google_patent_search_result = models.ForeignKey(GooglePatentSearchResult,
            blank=True, null=True, on_delete=models.CASCADE)
    resolution = models.IntegerField(
                choices=resolution_vals.choices(),
                default=resolution_vals.UNRESOLVED,
                )
    score = models.FloatField()
    evidence = models.TextField()
    SCORE_UNKNOWN = 1e99

    class Meta:
        unique_together = [['search', 'patent']]

    @property
    def resolution_text(self):
        return self.resolution_vals.get('label', self.resolution)

    @property
    def num_drug_evidence(self):
        import json
        return len(json.loads(self.evidence).get('drug_ev', []))

    @property
    def num_disease_evidence(self):
        import json
        return len(json.loads(self.evidence).get('disease_ev', []))

    @property
    def formatted_score(self):
        if self.score == self.SCORE_UNKNOWN:
            return 'N/A'
        else:
            return '%.2f' % self.score

