
import pytest
from mock import patch

from dtk.tests import make_ws, mock_dpi, mock_ppi, mock_pathways, tmpdir_path_helper

def simple_sim(sm1, sm2):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    m1 = Chem.MolFromSmiles(sm1)
    m2 = Chem.MolFromSmiles(sm2)
    fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, 2, nBits=1024)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, 2, nBits=1024)
    from rdkit import DataStructs
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def test_struct(tmp_path):
    a1 = tmp_path / "attrs.tsv"
    data = [
        ['id', 'attr', 'val'],
        ['D1', 'std_smiles', 'CCCC'],
        ['D2', 'std_smiles', 'OCO'],
        ['D3', 'std_smiles', 'ONO'],
        ['D4', 'std_smiles', 'CCCCCC'],
    ]
    with open(str(a1), 'w') as f:
        for line in data:
            f.write('\t'.join(line) + '\n')
    
    from dtk.metasim import StructSim
    fn_prefix = str(tmp_path/"simcache")
    StructSim.precompute_to_files(fn_prefix, [str(a1)])

    with patch('dtk.metasim.StructSim.get_paths') as get_paths:
        get_paths.return_value = (fn_prefix+'.fpdb.h5', fn_prefix+'.struct_metadata.json')
        ss = StructSim(fn_prefix)
        
    all_sms = ['CCCC', 'OCO', 'ONO', 'CCCCCC']

    # Normally this is used to remap back to agent IDs, but here we'll just map back to the original smiles.
    key_to_outputs = {sm: set([sm]) for sm in all_sms}

    for key in all_sms:
        assert ss.similar_to_key([key], 0.9, key_to_outputs) == {key: (1.0, key)}
    

    sim = ss.similar_to_key(['OCO'], 0.0, key_to_outputs)
    sim = {k: v[0] for k, v in sim.items()} # Strip info about which smiles it connected via
    expected = {sm:simple_sim('OCO', sm) for sm in all_sms}
    assert sim == pytest.approx(expected)

    sim = ss.similar_to_key(['OCO'], 0.4, key_to_outputs)
    sim = {k: v[0] for k, v in sim.items()} # Strip info about which smiles it connected via
    expected = {sm:simple_sim('OCO', sm) for sm in all_sms}
    assert sim == pytest.approx({k:v for (k,v) in expected.items() if v >= 0.4})


def test_dir_jac():
    agent_to_key = {
        1: ('P1', 'P2', 'P3'),
        2: ('P2', 'P3'),
        3: ('P1', 'P3'),
    }
    
    from dtk.metasim import DirectTargetSim
    dirsim = DirectTargetSim()
    output_map = dirsim.setup_output_map(agent_to_key, [], None)
    sims = dirsim.similar_to_key(('P1',), 0, output_map)
    assert sims == pytest.approx({
        1: 1/3,
        3: 1/2,
        })

    sims = dirsim.similar_to_key(('P3',), 0, output_map)
    assert sims == pytest.approx({
        1: 1/3,
        2: 1/2,
        3: 1/2,
        })

    sims = dirsim.similar_to_key(('P4',), 0, output_map)
    assert sims == pytest.approx({})

def test_metasim(
                tmp_path,
                make_ws,
                mock_dpi, # We need some fake DPI
                mock_ppi, # We need some fake PPI
                mock_pathways, # We need some fake pathways
                ):
    from dtk.metasim import MetaSim, get_sim_methods
    smiles = ['CCCC', 'CC(C(=O)O)N', 'CCCCCC', 'CC(C(=O)O)S', '[cH-]1cccc1']
    ws_attrs = []
    for i, smiles in zip(range(1, 6), smiles):
        ws_attrs += [
            (f'DB0{i}', 'canonical', f'Drug{i}'),
            (f'DB0{i}', 'm_dpimerge_id', f'DB0{i}'),
            (f'DB0{i}', 'std_smiles', smiles),
            ]
    ws = make_ws(ws_attrs)
    # Mock out DPI - name is important, needs dpimerge at the start.
    dpi_name='dpimerge.fake_dpi'
    dpi = [
        ('dpimerge_id', 'uniprot_id', 'evidence', 'direction'),
        ('DB01', 'P01', '0.5', '0'),
        ('DB01', 'P02', '0.9', '0'),
        ('DB02', 'P02', '1.0', '0'),
        ('DB03', 'P02', '0.5', '0'),
        ('DB04', 'P02', '0.5', '0'),
        ('DB04', 'P04', '0.5', '0'),
        ('DB05', 'P04', '0.5', '0'),
        ]
    mock_dpi(dpi_name, dpi)

    # Mock out PPI
    ppi = [
        ('prot1', 'prot2', 'evidence', 'direction'),
        ('P01', 'P01', '0.9', '0'),
        ('P02', 'P02', '0.9', '0'),
        ('P01', 'P02', '0.9', '0'),
        ('P03', 'P02', '0.9', '0'),
        ('P09', 'P02', '1.0', '0'),
        ('P01', 'P04', '0.9', '0'),
        ('P08', 'P04', '0.9', '0'),
        ]
    mock_ppi('ppi.fake_ppi', ppi)

    mock_pathways([
        ['PWY1', ['P01', 'P02']],
        ['PWY2', ['P01', 'P04']],
        ['PWY3', ['P01']],
        ['PWY4', ['P01', 'P02', 'P04']],
        ['PWY5', [f'P{i:02}' for i in range(99)]],
    ])


    from drugs.models import Drug
    all_agents = list(Drug.objects.all().values_list('id', flat=True))
    ref_agents = all_agents[:2]
    ws_agents = all_agents[2:]

    ref_sim_keys = MetaSim.make_similarity_keys(
        agents=ref_agents,
        methods=get_sim_methods(),
        dpi_choice=dpi_name,
        )
    print(f"Ref keys: {ref_sim_keys}")
    assert 'StructKey' in ref_sim_keys
    assert 'TargetsKey' in ref_sim_keys

    ws_sim_keys = MetaSim.make_similarity_keys(
        agents=ws_agents,
        methods=get_sim_methods(),
        dpi_choice=dpi_name,
        )
    print(f"WS keys: {ws_sim_keys}")

    a1 = tmp_path / "smiles.tsv"
    data = [
        ['id', 'attr', 'val'],
    ] + ws_attrs
    with open(str(a1), 'w') as f:
        for line in data:
            f.write('\t'.join(line) + '\n')
    
    from dtk.metasim import StructSim
    fn_prefix = str(tmp_path/"simcache")
    StructSim.precompute_to_files(fn_prefix, [str(a1)])

    with patch('dtk.metasim.StructSim.get_paths') as get_paths:
        get_paths.return_value = (fn_prefix+'.fpdb.h5', fn_prefix+'.struct_metadata.json')
        thresholds = {
            'rdkit': 0,
            'dirJac': 0,
            'indJac': 0,
            'pathway': 0,
            'prMax': 0,
            'indigo': 0,
        }

        from dtk.d2ps import D2ps
        ms = MetaSim(
            thresholds=thresholds,
            sim_choice='simdb',
            ppi_choice='ppi.fake_ppi',
            std_gene_list_set='fake_pathways',
            d2ps_method=D2ps.default_method,
            d2ps_threshold=0.3,
        )

        out = ms.run_all_keys(
            ref_sim_keys,
            ws_sim_keys,
            methods=[x.name for x in get_sim_methods()],
            precomputed={},
            cores=1,
            )




def test_indigo():
    # Note that these similarity values are simply recording what this currently outputs,
    # no math has been done to verify that it is a correct output, we're simply checking
    # that we don't inadvertently change it.
    from dtk.metasim import IndigoSim
    sim = IndigoSim()

    agent_to_key = {
        1: ['CCCC', 'CCCCCC'],
        2: ['CC(C(=O)O)N'],
        3: ['CC(C(=O)O)S'],
    }

    precmp = sim.setup_output_map(agent_to_key, None, None)

    out = sim.similar_to_key(['CCCC', 'CCCCCC'], 0, precmp)
    out = {k: v[0] for k, v in out.items()} # strip connecting smiles
    assert out == pytest.approx({1: 1.0, 2: 0.13636364, 3: 0.13636364})

    par_test = [['[cH-]1cccc1']]
    from dtk.parallel import pmap
    out = list(pmap(sim.similar_to_key, par_test, static_args={'threshold': 0, 'pre_cmp': precmp}))
    assert len(out) == 1
    out = [{k: v[0] for k, v in sim.items()} for sim in out]
    assert out[0] == pytest.approx({1: 0.1111111119, 2: 0.03846153989, 3: 0.03846153989})


def test_prsim(mock_ppi):
    # Note that these similarity values are simply recording what this currently outputs,
    # no math has been done to verify that it is a correct output, we're simply checking
    # that we don't inadvertently change it.

    # Mock out PPI
    ppi = [
        ('prot1', 'prot2', 'evidence', 'direction'),
        ('P01', 'P01', '0.9', '0'),
        ('P02', 'P02', '0.9', '0'),
        ('P01', 'P02', '0.9', '0'),
        ('P03', 'P02', '0.9', '0'),
        ('P09', 'P02', '1.0', '0'),
        ('P01', 'P04', '0.9', '0'),
        ('P08', 'P04', '0.9', '0'),
        ]
    mock_ppi('ppi.fake_ppi', ppi)
    agent_to_key = {
        1: ('P1', 'P2', 'P3'),
        2: ('P2', 'P3'),
        3: ('P1', 'P3'),
        4: ('P4', ),
        7: tuple(),
    }

    ref_to_key = {
        3: ('P1', 'P3'),
        5: ('P5', ),
        6: ('P2', ),
        7: tuple(),
    }
    
    from dtk.metasim import DiffusedTargetSim
    sim = DiffusedTargetSim('ppi.fake_ppi')

    ref_pre_cmp = sim.ref_precompute(ref_to_key)
    pre_cmp = sim.setup_output_map(agent_to_key, ref_to_key, ref_pre_cmp)
    pr = pre_cmp[0]
    from pprint import pprint
    pprint(pr.final_pr_d)

    sim1 = sim.similar_to_key(('P1', 'P3'), threshold=0, pre_cmp=pre_cmp)
    assert sim1 == {1: 1.0, 2: 1.0, 3: 1.0}

    sim2 = sim.similar_to_key(('P5', ), threshold=0, pre_cmp=pre_cmp)
    assert sim2 == {}

    sim3 = sim.similar_to_key(('P2', ), threshold=0, pre_cmp=pre_cmp)
    assert sim3 == {1: 1, 2: 1}

    sim4 = sim.similar_to_key(tuple(), threshold=0, pre_cmp=pre_cmp)
    assert sim4 == {}


def test_pathwaysim(mock_ppi, mock_pathways, tmpdir_path_helper):
    # Note that these similarity values are simply recording what this currently outputs,
    # no math has been done to verify that it is a correct output, we're simply checking
    # that we don't inadvertently change it.

    # Mock out PPI
    ppi = [
        ('prot1', 'prot2', 'evidence', 'direction'),
        ('P01', 'P01', '0.9', '0'),
        ('P02', 'P02', '0.9', '0'),
        ('P01', 'P02', '0.9', '0'),
        ('P03', 'P02', '0.9', '0'),
        ('P09', 'P02', '1.0', '0'),
        ('P01', 'P04', '0.9', '0'),
        ('P08', 'P04', '0.9', '0'),
        ]
    mock_ppi('ppi.fake_ppi', ppi)

    mock_pathways([
        ['PWY1', ['P01', 'P02']],
        ['PWY2', ['P01', 'P04']],
        ['PWY3', ['P01']],
        ['PWY4', ['P01', 'P02', 'P04']],
        ['PWY5', [f'P{i:02}' for i in range(99)]],
    ])
    agent_to_key = {
        1: ('P01', 'P02', 'P03'),
        2: ('P02', 'P03'),
        3: ('P01', 'P03'),
        4: ('P04', ),
        7: tuple(),
    }

    ref_to_key = {
        3: ('P01', 'P03'),
        5: ('P05', ),
        6: ('P02', ),
        7: tuple(),
    }
    
    from dtk.metasim import PathwaySim
    from dtk.d2ps import D2ps
    sim = PathwaySim(
        'ppi.fake_ppi',
        'fake.pathways',
        d2ps_method=D2ps.default_method,
        d2ps_threshold=0.3,
        )

    pre_cmp = sim.setup_output_map(agent_to_key, ref_to_key, None)

    sim1 = sim.similar_to_key(('P01', 'P03'), threshold=0, pre_cmp=pre_cmp)
    assert sim1 == pytest.approx({1: 0.97872, 2: 0.217391, 3: 1.0, 4: 0.217391}, abs=1e-4)

    sim2 = sim.similar_to_key(('P05', ), threshold=0, pre_cmp=pre_cmp)
    assert sim2 == {}

    sim3 = sim.similar_to_key(('P02', ), threshold=0, pre_cmp=pre_cmp)
    assert sim3 == pytest.approx({1: 0.212766, 3: 0.217391, 2: 1.0, 4: 0.25}, abs=1e-4)

    sim4 = sim.similar_to_key(tuple(), threshold=0, pre_cmp=pre_cmp)
    assert sim4 == {}
