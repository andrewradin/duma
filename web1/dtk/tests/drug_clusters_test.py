from dtk.drug_clusters import *
import unittest
class ClusterTest(unittest.TestCase):
    def test_drug_life_cycle(self):
        clr = Clusterer()
        key = ('chembl_id','CHEMBL123456')
        drug = clr.get_drug(key)
        self.assertEqual(drug.key,key)
        self.assertEqual(clr.get_drug(key),drug)
        prop = ('cas','12345-67-8')
        self.assertFalse( prop in drug.prop_set )
        self.assertFalse( prop in clr.drugs_by_prop )
        drug.add_prop( prop )
        self.assertTrue( prop in drug.prop_set )
        self.assertTrue( drug in clr.drugs_by_prop[prop] )
        drug.del_prop( prop )
        self.assertFalse( prop in drug.prop_set )
        self.assertFalse( prop in clr.drugs_by_prop )
        drug.add_prop( prop )
        self.assertTrue( prop in drug.prop_set )
        self.assertTrue( drug in clr.drugs_by_prop[prop] )
        self.assertEqual( clr.drugs_by_key[key], drug )
        drug.unhook()
        self.assertFalse( prop in clr.drugs_by_prop
                        and drug in clr.drugs_by_prop[prop]
                        )
        self.assertFalse( key in clr.drugs_by_key )
    def test_build_links(self):
        clr = Clusterer()
        keys = (
            ('coll_id','key1'),
            ('coll_id','key2'),
            ('coll_id','key3'),
            )
        drugs = [ clr.get_drug(k) for k in keys ]
        prop1 = ('type1','val1')
        prop2 = ('type2','val2')
        for d in drugs:
            d.add_prop(prop1)
        drugs[0].add_prop(prop2)
        drugs[1].add_prop(prop2)
        drug = drugs[0]
        drug.build_links()
        self.assertFalse( drugs[0] in drug.links )
        self.assertTrue( drugs[1] in drug.links )
        self.assertTrue( drugs[2] in drug.links )
        self.assertEqual( set([prop1,prop2]), drug.links[drugs[1]] )
        self.assertEqual( set([prop1]), drug.links[drugs[2]] )
        self.assertEqual( set(drugs), drug.get_cluster_as_set() )
        drugs.append( clr.get_drug( ('another_collection','another_key') ) )
        prop3 = ('type2','val3')
        drugs[2].add_prop(prop3)
        drugs[3].add_prop(prop3)
        clr.build_links()
        self.assertEqual( set(drugs), drug.get_cluster_as_set() )
        root = drug.get_cluster_as_tree()
        self.assertEqual( '\n'+root.pretty_print(),
'''
Drug(coll_id,key1)
  type1:val1|type2:val2->
    Drug(coll_id,key2)
      type1:val1->
  type1:val1->
    Drug(coll_id,key3)
      type2:val3->
        Drug(another_collection,another_key)
'''
                        )
    def test_clustering(self):
        clr = Clusterer()
        keys = (
            ('coll_id','key1'),
            ('coll_id','key2'),
            ('coll_id','key3'),
            ('another','key4'),
            )
        drugs = [ clr.get_drug(k) for k in keys ]
        prop1 = ('type1','val1')
        prop2 = ('type2','val2')
        prop3 = ('type3','val3')
        drugs[0].add_prop(prop1)
        drugs[1].add_prop(prop2)
        drugs[2].add_prop(prop2)
        drugs[2].add_prop(prop3)
        drugs[3].add_prop(prop3)
        clr.build_links()
        stats = clr.link_stats()
        self.assertEqual( stats.values[0],3 ) # 3 props
        self.assertEqual( stats.children[0].values[0],1 ) # 1 unique
        self.assertEqual( stats.children[1].values[0],2 ) # 2 shared by 2 drugs
        clr.trim_props()
        # unique prop should be gone
        stats = clr.link_stats()
        self.assertEqual( stats.values[0],2 ) # 2 props
        self.assertEqual( stats.children[0].values[0],2 ) # 2 shared by 2 drugs
        # test drug trimming
        self.assertEqual( len(clr.drugs_by_key), 4 )
        clr.trim_disconnected_drugs()
        self.assertEqual( len(clr.drugs_by_key), 3 )
        # test clustering
        clr.build_links()
        clr.build_clusters()
        self.assertEqual( len(clr.clusters), 1 )
        self.assertFalse( drugs[0] in clr.clusters[0] )
        self.assertTrue( drugs[1] in clr.clusters[0] )
        self.assertTrue( drugs[2] in clr.clusters[0] )
        self.assertTrue( drugs[3] in clr.clusters[0] )




def test_rebuilt_cluster(tmp_path):
    cluster_file = tmp_path / 'clusters.tsv'
    samples = [
        ['bindingdb_id', 'BDBM001', 'chembl_id', 'CHEMBL003'],
        ['duma_id', 'DUMA002'],
        ['duma_id', 'DUMA003', 'chembl_id', 'CHEMBL004', 'ncats_id', 'NCATS5'],
        ['duma_id', 'DUMA006', 'duma_id', 'DUMA009', 'ncats_id', 'NCATS8'],
        ]
    lines = ('\t'.join(sample) for sample in samples)
    cluster_file.write_text('\n'.join(lines))
    cluster_fn = str(cluster_file)


    from mock import patch
    with patch('dtk.drug_clusters.RebuiltCluster.get_cluster_file_path',
                    side_effect=lambda *args, **kwargs: cluster_fn,
                    autospec=True):
        from dtk.drug_clusters import RebuiltCluster
        rc = RebuiltCluster(base_key=('bindingdb_id', 'BDBM001'), version=1)
        assert dict(rc.drug_keys) == {'bindingdb_id': 'BDBM001', 'chembl_id': 'CHEMBL003'}

        # BDBM002 isn't in the clusters file, it should return itself.
        rc = RebuiltCluster(base_key=('bindingdb_id', 'BDBM002'), version=1)
        assert dict(rc.drug_keys) == {'bindingdb_id': 'BDBM002'}

        # Try a bulk load.
        keys = [
                ('duma_id', 'DUMA002'),
                ('ncats_id', 'NCATS5'),
                ('bad_id', 'BAD01'),
                ('chembl_id', 'CHEMBL004'),
                ]
        clusters = RebuiltCluster.load_many_clusters(1, keys)
        assert len(clusters) == len(keys)
        assert dict(clusters[0].drug_keys) == {'duma_id': 'DUMA002'}
        assert dict(clusters[1].drug_keys) == {
                'duma_id': 'DUMA003',
                'chembl_id': 'CHEMBL004',
                'ncats_id': 'NCATS5',
                }
        assert dict(clusters[2].drug_keys) == {'bad_id': 'BAD01'}
        assert clusters[3].drug_keys == clusters[1].drug_keys

        # look up duma id info for a cluster
        # - no duma id case
        rc = RebuiltCluster(base_key=('bindingdb_id', 'BDBM001'), version=1)
        assert rc.best_duma_key() == None
        # - single duma id case
        rc = RebuiltCluster(base_key=('ncats_id', 'NCATS5'), version=1)
        assert rc.best_duma_key() == 'DUMA003'
        # - multiple duma id case
        rc = RebuiltCluster(base_key=('ncats_id', 'NCATS8'), version=1)
        assert rc.best_duma_key() == 'DUMA009'

