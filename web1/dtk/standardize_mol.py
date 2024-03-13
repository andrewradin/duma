#!/usr/bin/env python3

# We need a modern rdkit.
import dtk.rdkit_2019

from rdkit import Chem, RDLogger

# Without this the cleanup process is very verbose.
RDLogger.DisableLog('rdApp.info')

# Also, without this the tautomerization stuff is extra slow
rdlg = RDLogger.logger()
rdlg.setLevel(RDLogger.WARNING)
import logging
logging.getLogger("rdkit.Chem.MolStandardize").setLevel(logging.WARN)


def base_standardize_mol(mol):
    """Standardizes a molecule.

    The idea is that this will eliminate trivial/cosmetic differences in
    molecules that, for our purposes, we consider to be the same.

    See https://github.com/chembl/ChEMBL_Structure_Pipeline

    The primary difference with ChEMBL's pipeline is in fragment handling.
    Rather than just picking the largest fragment, it removes only fragments
    that are known salts/solvents.
    """
    from chembl_structure_pipeline import standardizer
    std_mol = standardizer.standardize_mol(mol)
    if not std_mol:
        return std_mol
    std_mol, _ = standardizer.get_parent_mol(std_mol)
    if not std_mol:
        return std_mol
    # Some molecules crash with uninitialized rings if you don't do SSSR.
    Chem.GetSSSR(std_mol)
    return std_mol


def old_base_standardize_mol(mol):
    """Standardizes a molecule.

    The idea is that this will eliminate trivial/cosmetic differences in
    molecules that, for our purposes, we consider to be the same.

    Tautomer canonicalization is separate because it is slower.
    """
    # See https://github.com/rdkit/rdkit/blob/master/Docs/Notebooks/MolStandardize.ipynb
    # as a reference for methods available.
    from rdkit.Chem.MolStandardize import rdMolStandardize
    fragment_chooser = rdMolStandardize.LargestFragmentChooser()
    uncharger = rdMolStandardize.Uncharger()

    mol = rdMolStandardize.Cleanup(mol)
    mol = fragment_chooser.choose(mol)
    mol = uncharger.uncharge(mol)
    return mol

def standardize_mol_tautomer(mol, max_tautomers=1000):
    """Picks a canonical tautomer from the various possible tautomers.
    
    Be careful not to set max_tautomers too high, apparently we can run out of memory doing this.
    """
    # This is the newly ported tautomer code for more recent rdkits.
    # It seems to be wildly faster than the old method, and produces
    # better results.
    from rdkit.Chem.MolStandardize import rdMolStandardize
    enumerator = rdMolStandardize.TautomerEnumerator()
    enumerator.SetMaxTautomers(max_tautomers)
    #enumerator.SetReassignStereo(False)
    #enumerator.SetRemoveBondStereo(False)
    #enumerator.SetRemoveSp3Stereo(False)
    if max_tautomers > 2000:
        # Enumeration stops when hitting max tautomers or transforms, usually
        # transforms hits first.
        # Could expose another setting for this instead.
        enumerator.SetMaxTransforms(max_tautomers)
    return enumerator.Canonicalize(mol)


def old_standardize_mol_tautomer(mol, max_tautomers=1000):
    # Old method from before this was ported into rdMolStandardize.
    # Keeping this around for a bit just in case we need to refer to it.
    from rdkit.Chem.MolStandardize.tautomer import TautomerCanonicalizer
    tautomer = TautomerCanonicalizer(max_tautomers=max_tautomers)
    return tautomer.canonicalize(mol)


def standardize_mol(mol, max_tautomers=1000):
    return standardize_mol_tautomer(base_standardize_mol(mol),
                                    max_tautomers=max_tautomers)


if __name__ == "__main__":
    import sys
    for line in sys.stdin:
        try:
            m = Chem.MolFromSmiles(line.strip())
            sm = standardize_mol(m)
            print(Chem.MolToSmiles(sm))
        except:
            import traceback as tb
            tb.print_exc()
