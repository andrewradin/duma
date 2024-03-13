import pytest
from mock import patch

def test_label_from_filename():
    from dtk.prot_map import DpiMapping
    legacy_label = 'mykey.mydetail'
    legacy_fn = f'dpi.{legacy_label}.tsv'
    assert DpiMapping.label_from_filename(legacy_fn) == legacy_label
    new_label = 'mydetail.v1'
    new_fn = f'matching.{new_label}.dpimerge.tsv'
    assert DpiMapping.label_from_filename(new_fn) == new_label

def test_dpi_choices(db):
    from dtk.prot_map import DpiMapping
    with patch.object(DpiMapping, 'dpi_names', autospec=True) as dpi_names:
        dpi_names.return_value = [
            'dpi.v1',
            'dpi.v2',
            'dpi.v3',
            'dpi-moa.v2',
            'dpi+Advil.v2',
            'uniprot.v1',
            'chembl',
        ]

        # Nothing uploaded
        choices = DpiMapping.choices()
        expected = [
            ('Legacy', [('chembl', 'chembl')]),
        ]
        assert choices == expected, "Nothing uploaded, no versioned should show"

        # v1 uploaded only
        from drugs.models import DpiMergeKey, Drug, Collection
        coll = Collection.objects.create()
        drug = Drug.objects.create(collection=coll)
        DpiMergeKey.objects.create(version=1, drug=drug)
        choices = DpiMapping.choices()
        expected = [
            ('Latest', [('dpi.v1', 'dpi.v1'), ('uniprot.v1', 'uniprot.v1')]),
            ('Uniprot', [('uniprot.v1', 'uniprot.v1')]),
            ('Legacy', [('chembl', 'chembl')]),
        ]
        assert choices == expected, "V1 only"

        # v2 uploaded
        DpiMergeKey.objects.create(version=2, drug=drug)
        choices = DpiMapping.choices()
        expected = [
            ('Latest', [('dpi.v2', 'dpi.v2'), ('dpi-moa.v2', 'dpi-moa.v2'),]),
            ('MoA', [('dpi-moa.v2', 'dpi-moa.v2')]),
            ('Uniprot', [('uniprot.v1', 'uniprot.v1')]),
            ('Combo', [('dpi+Advil.v2', 'dpi+Advil.v2')]),
            ('Versioned', [('dpi.v1', 'dpi.v1')]),
            ('Legacy', [('chembl', 'chembl')]),
        ]
        assert choices == expected, "Up to v2"