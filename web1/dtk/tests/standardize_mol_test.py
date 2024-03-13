
import pytest

from dtk.standardize_mol import standardize_mol, base_standardize_mol, standardize_mol_tautomer
from rdkit import Chem


from pytest import mark

def uncharge(mol):
    from rdkit.Chem.MolStandardize import rdMolStandardize
    fragment_chooser = rdMolStandardize.LargestFragmentChooser()
    uncharger = rdMolStandardize.Uncharger()

    mol = uncharger.uncharge(mol)

    return mol

def standardize(smile):
    if smile.startswith('In'):
        std_mol = Chem.MolFromInchi(smile)
    else:
        std_mol = Chem.MolFromSmiles(smile)
        # This mimicks make_std_smiles.
        std_mol = Chem.MolFromInchi(Chem.MolToInchi(std_mol))
    
    mol = standardize_mol(std_mol)
    return Chem.MolToSmiles(mol, isomericSmiles=False)

def check_mols_same(mols):
    err = []
    for name, *smiles in mols:
        print("Doing ", name, smiles)
        std_smiles = set()
        for smile in smiles:
            std_smiles.add(standardize(smile))
        
        if len(std_smiles) != 1:
            msg = '\n'.join(std_smiles)
            err.append(f"\n*Multiple std_smiles for {name}*: \n{msg}\n")
        else:
            print(f"{name} all converts to same smiles {next(iter(std_smiles))}")
    
    fail = len(err) != 0
    err.append(f"Failures: {len(err)} / {len(mols)}")
    assert not fail,  '\n'.join(err)


def check_mols_different(mols):
    err = []
    for name, *smiles in mols:
        print("Doing ", name, smiles)
        std_smiles = set()
        for smile in smiles:
            std_smiles.add(standardize(smile))
        
        if len(std_smiles) != len(smiles):
            msg = '\n'.join(std_smiles)
            err.append(f"\n*Undesired overlapping std_smiles for {name}*: \n{msg}\n")
        else:
            print(f"{name} all converts to different smiles")
    
    fail = len(err) != 0
    err.append(f"Failures: {len(err)} / {len(mols)}")
    assert not fail,  '\n'.join(err)

def test_salts():
    mols = [
        # Chloride, Sulate, and a base form.
        [
            "Berberine",
            "COC1=C(C2=C[N+]3=C(C=C2C=C1)C4=CC5=C(C=C4CC3)OCO5)OC.[Cl-]",
            "COC1=C(C2=C[N+]3=C(C=C2C=C1)C4=CC5=C(C=C4CC3)OCO5)OC.OS(=O)(=O)[O-]",
            "COC1=C(C2=C[N+]3=C(C=C2C=C1)C4=CC5=C(C=C4CC3)OCO5)OC",
        ],
        # Have had salt issues here (propionate).
        [
            "Drostanolone",
            # These are both non-priopionate forms.
            "[H][C@@]12CC[C@H](O)[C@@]1(C)CC[C@@]1([H])[C@@]2([H])CC[C@@]2([H])CC(=O)[C@H](C)C[C@]12C",
            "C[C@@H]1C[C@]2([C@@H](CC[C@@H]3[C@@H]2CC[C@]4([C@H]3CC[C@@H]4O)C)CC1=O)C",
            # Has the propionate attached; our standardization doesn't currently handle this.
            #"[H][C@@]12CC[C@H](OC(=O)CC)[C@@]1(C)CC[C@@]1([H])[C@@]2([H])CC[C@@]2([H])CC(=O)[C@H](C)C[C@]12C",
            #"CCC(=O)O[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC[C@@H]4[C@@]3(C[C@H](C(=O)C4)C)C)C",
        ],
        # Includes a sodium salt and lots of slash characters.
        [
            "Acitretin",
            r"COC1=C(C)C(C)=C(\C=C\C(\C)=C\C=C\C(\C)=C\C(O)=O)C(C)=C1",
            r"COc1cc(C)c(/C=C/C(C)=C/C=C/C(C)=C/C(=O)O)c(C)c1C",
            r"CC1=CC(=C(C(=C1/C=C/C(=C/C=C/C(=C/C(=O)O)/C)/C)C)C)OC",
            r"CC1=CC(=C(C(=C1/C=C/C(=C/C=C/C(=C/C(=O)[O-])/C)/C)C)C)OC.[Na+]",
        ],
        # Hydrates
        # This also includes some dexpramipexole which is an enantiomer.
        [
            "Pramipexole",
            "CCCNC1CCC2=C(C1)SC(=N2)N.O.Cl.Cl",
            "CCCNC1CCC2=C(C1)SC(=N2)N.Cl.Cl",
            "CCCN[C@H]1CCC2=C(C1)SC(N)=N2",
            "CCCNC1CCc2nc(N)sc2C1",
            "CCCN[C@@H]1CCC2=C(C1)SC(N)=N2",
        ],
    ]
    check_mols_same(mols)


def test_tautomers():
    mols = [
        [
            "CHEMBL17139, CHEMBL1907917, CHEMBL17553",
            "CCN(C(=O)C1=CCCC1C(=O)NCc1ccc(C(=N)N)cc1)c1cccc(OC)c1",
            "CCN(C(=O)C1CCC=C1C(=O)NCc1ccc(C(=N)N)cc1)c1cccc(OC)c1",
            "CCN(C(=O)C1=CCC[C@@H]1C(=O)NCc1ccc(C(=N)N)cc1)c1cccc(OC)c1",
        ],
    ]
    check_mols_same(mols)

def test_same_mol():
    # First entry is name, all others SMILES codes from various sources that should all
    # be the same molecule after standardization.
    mols = [
        # Aromaticity/kekulization can lead to different double bonds / orientations.
        [
            "MLN-8054",
            "O=C(O)c1ccc(Nc2ncc3c(n2)-c2ccc(Cl)cc2C(c2c(F)cccc2F)=NC3)cc1",
            "OC(=O)c1ccc(Nc2ncc3CN=C(c4cc(Cl)ccc4-c3n2)c2c(F)cccc2F)cc1",
            "OC(=O)C1=CC=C(NC2=NC=C3CN=C(C4=C(C=CC(Cl)=C4)C3=N2)C2=C(F)C=CC=C2F)C=C1",
            "C1C2=CN=C(N=C2C3=C(C=C(C=C3)Cl)C(=N1)C4=C(C=CC=C4F)F)NC5=CC=C(C=C5)C(=O)O",
        ],
        # There are two places that differ in double bonds depending on std method.
        # This one might be a tautomerizaztion problem, but there are too many tautomers to enumerate (OOM trying to do them all)
        # If we round-trip through INCHI (or start with it) though, we do match up.
        [
            "Livoletide",
            # DB15188
            "[H][C@@]12CCCN1C(=O)[C@H](CO)NC(=O)[C@H](CCC(N)=O)NC(=O)[C@@H](NC(=O)[C@H](CCCNC(N)=N)NC(=O)[C@H](CCC(N)=O)NC(=O)[C@H](CC1=CN=CN1)NC(=O)[C@H](CCC(O)=O)NC2=O)C(C)C",
            # PUBCHEM57341282
            "CC(C)[C@H]1C(=O)N[C@H](C(=O)N[C@H](C(=O)N2CCC[C@H]2C(=O)N[C@H](C(=O)N[C@H](C(=O)N[C@H](C(=O)N[C@H](C(=O)N1)CCCN=C(N)N)CCC(=O)N)CC3=CN=CN3)CCC(=O)O)CO)CCC(=O)N",
            # CHEMBL2029605
            "CC(C)[C@@H]1NC(=O)[C@H](CCCNC(=N)N)NC(=O)[C@H](CCC(N)=O)NC(=O)[C@H](Cc2cnc[nH]2)NC(=O)[C@H](CCC(=O)O)NC(=O)[C@@H]2CCCN2C(=O)[C@H](CO)NC(=O)[C@H](CCC(N)=O)NC1=O",
        ],
        # Non-aromatic-related double bond O vs single bond OH.
        # Is drugbank just wrong here?  Seems like yes.
        # PubChem has 3 different isomeric forms of Freselestat, and then also separately has an entry for drugbank's molecule.
        [
            "Freselestat",
            # This is the drugbank molecule
            #"CC(C)[C@@H](NC(=O)CN1C(=O)C(N)=CN=C1C1=CC=CC=C1)[C@@H](O)C1=NN=C(O1)C(C)(C)C",
            # This is the pubchem molecule referencing the drugbank one.
            #"CC(C)[C@H]([C@H](C1=NN=C(O1)C(C)(C)C)O)NC(=O)CN2C(=NC=C(C2=O)N)C3=CC=CC=C3",
            "CC(C)C(NC(=O)Cn1c(-c2ccccc2)ncc(N)c1=O)C(=O)c1nnc(C(C)(C)C)o1",
            "CC(C)[C@H](NC(=O)Cn1c(-c2ccccc2)ncc(N)c1=O)C(=O)c1nnc(C(C)(C)C)o1",
            "CC(C)[C@@H](NC(=O)Cn1c(ncc(N)c1=O)-c1ccccc1)C(=O)c1nnc(o1)C(C)(C)C	",
        ],
        # The two CHEMBL ones are left and right hand versions of the same molecule.
        # The 3rd SMILES is from bindingdb, and that's the outlier.
        [
            "CHEMBL110898",
            "CC(C)(C)OC(=O)N[C@H](C(=O)N1CCC[C@H]1C(=O)NC(C=O)CCCN=C(N)N)c1cccc2ccccc12",
            "CC(C)(C)OC(=O)N[C@@H](C(=O)N1CCC[C@H]1C(=O)NC(C=O)CCCN=C(N)N)c1cccc2ccccc12",
            "CC(C)(C)OC(=O)N[C@@H](C(=O)N1CCC[C@H]1C(=O)NC(CCCNC(N)=N)C=O)c1cccc2ccccc12",
        ]
    ]
    check_mols_same(mols)



def test_isomers():
    # Technically there are isomers in a few others as well.
    # For now we aren't generating isomeric smiles and so these should all get condensed.
    mols =[
        [
            "Bisindolylmaleimide II",
            "[H]N1C=C(C2=CC=CC=C12)C1=C(C(=O)N([H])C1=O)C1=CN(CC[C@H]2CCCN2C)C2=CC=CC=C12",
            "[H]N1C=C(C2=CC=CC=C12)C1=C(C(=O)N([H])C1=O)C1=CN(CC[C@@H]2CCCN2C)C2=CC=CC=C12",
            "CN1CCCC1CCn1cc(C2=C(c3c[nH]c4ccccc34)C(=O)NC2=O)c2ccccc21",
            "CN1CCCC1CCN2C=C(C3=CC=CC=C32)C4=C(C(=O)NC4=O)C5=CNC6=CC=CC=C65",
            "CN1CCC[C@H]1CCN2C=C(C3=CC=CC=C32)C4=C(C(=O)NC4=O)C5=CNC6=CC=CC=C65",
            "CN1CCC[C@@H]1CCN2C=C(C3=CC=CC=C32)C4=C(C(=O)NC4=O)C5=CNC6=CC=CC=C65",
        ]
    ]
    check_mols_same(mols)


def test_metal_fragments():
    # These are all molecules we used to standardize to "S" in our original
    # molecule standardization, due to fragmenting.
    orig_to_std = [
        # Most of these we don't _really_ care about the specifics, just
        # want to make sure they're not becoming degenerate.
        ["[Cu]12[Cu]3[Cu]4[Cu]1S234", "[Cu].[Cu].[Cu].[Cu].[S]"],
        ["S[Mo](=O)=O", "[Mo+].[O].[O].[SH-]"],
        ["[SH-].[Cu].[Cu+]", "[Cu+].[Cu].[SH-]"],
        ["O=[Mo]=O.[SH-]", "S.[Mo].[O].[O]"],
        ["[SH-].[SH-].S=[Mo]=S", "S.S.[Mo].[S].[S]"],
        ["[SH-].[Cu].[Cu].[Cu].[Cu+]", "[Cu+].[Cu].[Cu].[Cu].[SH-]"],
        ["[SH-].[Au].[Au+]", "[Au+].[Au].[SH-]"],

        # Everything below here still converts to "S".
        ["S", "S"],
        ["[SH-]", "S"],
        ["[32SH2]", "S"],
        ["[33SH2]", "S"],
        ["[34SH2]", "S"],
        # NOTE: Everything below here converts to "S" but is questionable
        # at best, some probably wrong.
        ["[SH-].[Ag].[Ag+]", "S"],
        ["[Li+].[Li+].[SH-]", "S"],
        ["[7Li][7Li].S", "S"],
        ["[Na+].[SH-]", "S"],
        ["[SH-].[K+]", "S"],
        ["[Na+].[Na+].[SH-]", "S"],
    ]

    errs = []
    for orig, exp_std in orig_to_std:
        std = standardize(orig)

        if std != exp_std:
            errs.append(f'Expected {std} == {exp_std} for {orig}')
    

    fail = len(errs) > 0
    assert not fail, '\n' + '\n'.join(errs)

def test_mixtures():
    # Some drugs are mixtures of other drugs.
    # How we handle those is fairly arbitrary.
    # A good example of this is Aminophylline (https://go.drugbank.com/drugs/DB01223), which is just a mixture of:
    # 2 Theophylline molecules (https://go.drugbank.com/drugs/DB00277)
    # 1 Ethylenediamine molecule (https://go.drugbank.com/drugs/DB14189)
    #
    # There are also other drugs that are mixtures of theophylline and other small molecules, such as
    # Ambuphylline (https://go.drugbank.com/drugs/DB13812)

    mols =[
        [
            "Various Theophylline mixtures",
            "Cn1c(=O)c2nc[nH]c2n(C)c1=O",
            "Cn1c2nc[nH]c2c(=O)n(C)c1=O",
            "CC(C)(N)CO.Cn1c(=O)c2[nH]cnc2n(C)c1=O",
            "Cn1c(=O)c2[nH]cnc2n(C)c1=O.Cn1c(=O)c2[nH]cnc2n(C)c1=O.NCCN",

        ]
    ]
    check_mols_same(mols)


def test_choline_and_salicylate():
    # Choline and salicylate both appears as salts and molecules of various forms.
    # Our old standardization tended to overreduce these, so we had a ton of things matching choline.
    # Chembl's standardization underreduces these because it sees everything as a salt, strips it all, and gives up.

    cholines = [
        "C[N+](C)(C)CCO",
        "[Cl-].C[N+](C)([11CH3])CCO",
        "C[N+](C)([11CH3])CCO",
    ]

    oxtriphylline = "C[N+](C)(C)CCO.CN1C2=C([N-]C=N2)C(=O)N(C)C1=O"
    chol_salicylate = "C[N+](C)(C)CCO.OC1=CC=CC=C1C([O-])=O"
    chol_mag_trisalicylate = "[Mg++].C[N+](C)(C)CCO.OC1=CC=CC=C1C([O-])=O.OC1=CC=CC=C1C([O-])=O.OC1=CC=CC=C1C([O-])=O"

    # Old pipeline can pass the first test; chembl pipeline can pass the second.
    if False:
        check_mols_same([
            ["Cholines"] + cholines,
            ["Salicylates", chol_salicylate, chol_mag_trisalicylate],
            ])
    else:
        check_mols_different([
            ["Non-choline 1", cholines[0], oxtriphylline],
            ["Non-choline 2", cholines[0], chol_salicylate],
            ["Non-choline 3", cholines[0], chol_mag_trisalicylate],
            ])

def test_serine():
    # Lots of forms of this; but also, serine vanadate, which shouldn't be the same thing.
    serines = [
        "N[C@@H](CO)C(O)=O",
        "N[C@H](CO)C(=O)O",
        "C([C@@H](C(=O)O)N)O.Cl",
        # The next smiles tautomerizes different to the above, which is an rdkit bug.
        # They have fixed it here - https://github.com/rdkit/rdkit/issues/3755
        # It should make it into the 2021.03 release.
        # "C(C(C(=O)O)N)O",
    ]

    # The old pipeline reduced this to serine, which is wrong.
    # New pipeline can't figure out they're the same, though.
    serine_vanadates = [
        "N[C@@H](CO[V](O)([O-])([O-])[O-])C(O)=O",
        "C([C@@H](C(=O)O)N)O.O.[OH-].[OH-].[OH-].[V]",
    ]


    check_mols_same([
        ["Serines"] + serines,
        #["Serine Vanadate"] + serine_vanadates,
    ])


def test_adenosine():
    # Some of these have Tungstun (W)... does that make it different?
    mols = [
        [
        "Adenosine",
        "NC1=NC=NC2=C1N=CN2[C@@H]1O[C@H](CO)[C@@H](O)[C@@H]1O",
        "NC1=C2N=CN([C@@H]3O[C@H](CO)[C@@H](O)[C@H]3O)C2=NC=N1",
        # These two have different amounts of Tungsten (W) in them, which makes a difference in the
        # ChEMBL pipeline.
        #"NC1=C2N=CN(C3OC(CO[W](O)(=O)O[W](O)(O)=O)C(O)C3O)C2=NC=N1",
        #"[H]N([H])C1=NC=NC2=C1N=CN2[C@]1([H])O[C@@]([H])(CO[W](O)(O)=O)[C@]([H])(O)[C@@]1([H])O",
        ],
    ]

    check_mols_same(mols)


def test_sucralfate():
    # https://go.drugbank.com/drugs/DB00364
    # https://go.drugbank.com/drugs/DB01901
    # Sucrosofate vs Sucralfate
    # Sucralfate is an aluminum salt of sucrosofate.
    # Pretty complex though there are 9 Al atoms in there and some water.
    # If you don't go via inchi, this fails with the chembl pipeline.

    sucro = "C([C@@H]1[C@H]([C@@H]([C@H]([C@H](O1)O[C@]2([C@H]([C@@H]([C@H](O2)COS(=O)(=O)O)OS(=O)(=O)O)OS(=O)(=O)O)COS(=O)(=O)O)OS(=O)(=O)O)OS(=O)(=O)O)OS(=O)(=O)O)OS(=O)(=O)O"
    sucro_db = "OS(=O)(=O)OC[C@H]1O[C@@](COS(O)(=O)=O)(O[C@H]2O[C@H](COS(O)(=O)=O)[C@@H](OS(O)(=O)=O)[C@H](OS(O)(=O)=O)[C@H]2OS(O)(=O)=O)[C@@H](OS(O)(=O)=O)[C@@H]1OS(O)(=O)=O"
    sucra = "C([C@@H]1[C@H]([C@@H]([C@H]([C@H](O1)O[C@]2([C@H]([C@@H]([C@H](O2)COS(=O)(=O)O[Al])OS(=O)(=O)O[Al])OS(=O)(=O)O[Al])COS(=O)(=O)O[Al])OS(=O)(=O)O[Al])OS(=O)(=O)O[Al])OS(=O)(=O)O[Al])OS(=O)(=O)O[Al].O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.O.[Al]"
    sucra_db = "O.O[Al](O)O.O[Al](O)OS(=O)(=O)OC[C@H]1O[C@@](COS(=O)(=O)O[Al](O)O)(O[C@H]2O[C@H](COS(=O)(=O)O[Al](O)O)[C@@H](OS(=O)(=O)O[Al](O)O)[C@H](OS(=O)(=O)O[Al](O)O)[C@H]2OS(=O)(=O)O[Al](O)O)[C@@H](OS(=O)(=O)O[Al](O)O)[C@@H]1OS(=O)(=O)O[Al](O)O"
    sucra_duma = "O=S(=O)(O)OC[C@H]1O[C@H](O[C@]2(COS(=O)(=O)O)O[C@H](COS(=O)(=O)O)[C@@H](OS(=O)(=O)O)[C@@H]2OS(=O)(=O)O)[C@H](OS(=O)(=O)O)[C@@H](OS(=O)(=O)O)[C@@H]1OS(=O)(=O)O"
    sucra_pc = "C([C@@H]1[C@H]([C@@H]([C@H]([C@H](O1)O[C@]2([C@H]([C@@H]([C@H](O2)COS(=O)(=O)O[Al])OS(=O)(=O)O[Al])OS(=O)(=O)O[Al])COS(=O)(=O)O[Al])OS(=O)(=O)O[Al])OS(=O)(=O)O[Al])OS(=O)(=O)O[Al])OS(=O)(=O)O[Al]"

    mols = [
        [ "Sucr", sucro, sucra, sucro_db, sucra_db, sucra_duma, sucra_pc ]
    ]

    check_mols_same(mols)


def test_thyroxine():
    # Thyroxine (T4) vs Thyronine (T3) vs Liotrix (T3 + T4 mixture)
    # Old pipeline marks Thyroxine and Liotrix the same, due to largest fragment chooser.
    # CHembl pipeline marks all 3 as different.
    mols = [
        [
            "Thyroxine vs Thyronine vs Liotrix (T3 + T4 mixture)",
            "N[C@@H](CC1=CC(I)=C(OC2=CC(I)=C(O)C(I)=C2)C(I)=C1)C(O)=O",
            "N[C@@H](CC1=CC(I)=C(OC2=CC(I)=C(O)C=C2)C(I)=C1)C(O)=O",
            "[Na+].[Na+].N[C@@H](CC1=CC(I)=C(OC2=CC(I)=C(O)C=C2)C(I)=C1)C([O-])=O.N[C@@H](CC1=CC(I)=C(OC2=CC(I)=C(O)C(I)=C2)C(I)=C1)C([O-])=O",
        ]
    ]

    check_mols_different(mols)

def test_lithium():
    # In the old pipeline, the lithium would be stripped.
    # The chembl pipeline instead keeps all fragments if they're all salts.
    orig_to_std = [
        # Lithium carbonate
        ["O=C([O-])[O-].[Li+].[Li+]", "O=C([O-])[O-].[Li+].[Li+]"],
        ["[Li+].[Li+].[O-]C([O-])=O", "O=C([O-])[O-].[Li+].[Li+]"],
        # Lithium citrate
        ["[Li+].[Li+].[Li+].OC(CC([O-])=O)(CC([O-])=O)C([O-])=O", "O=C([O-])CC(O)(CC(=O)[O-])C(=O)[O-].[Li+].[Li+].[Li+]"],
    ]

    errs = []
    for orig, exp_std in orig_to_std:
        std = standardize(orig)

        if std != exp_std:
            errs.append(f'Expected {std} == {exp_std} for {orig}')
    

    fail = len(errs) > 0
    assert not fail, '\n' + '\n'.join(errs)
