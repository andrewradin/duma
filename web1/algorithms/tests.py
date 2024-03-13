from django.test import TestCase

class PathsumTestCase(TestCase):
    def test_tissue_settings(self):
        from .run_path import get_tissue_ids_from_settings
        self.assertEqual(get_tissue_ids_from_settings({}),[])
        self.assertEqual(get_tissue_ids_from_settings({
                        'some_irrelevant_key':0.95,
                        }),[])
        self.assertEqual(get_tissue_ids_from_settings({
                        't_1':0.95,
                        }),[1])
        self.assertEqual(get_tissue_ids_from_settings({
                        't_2':0.95,
                        't_10':0.95,
                        't_9999':0.95,
                        }),[2,10,9999])
        from .run_path import get_tissue_settings_keys
        self.assertEqual(get_tissue_settings_keys(33),('t_33','t_33_fc'))

