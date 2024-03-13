### for commonly used drug-structure methods


# basically just an error handler wrapped around RDKit's code
def getCanonSmilesFromSmiles(smiles):
    import sys
    if sys.version_info[0] >= 3:
        import dtk.rdkit_2019
    # Load RDKit
    from rdkit import Chem
    assert Chem.inchi.INCHI_AVAILABLE,'wrong RDKit version; in venv?'
    try:
        m1 = Chem.MolFromSmiles(smiles)
        if m1 is None:
            return None
        return str(Chem.MolToSmiles(m1,isomericSmiles=True))
    except:
        return None

