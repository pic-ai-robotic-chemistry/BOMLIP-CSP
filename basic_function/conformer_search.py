from rdkit import Chem
from rdkit.Chem import AllChem
import os


def generate_conformers(molecule, num_conformers=10, max_attempts=1000, rms_thresh=0.2):
    """
    Generate molecular conformers.

    Parameters:
        molecule (RDKit Mol object): The input molecule.
        num_conformers (int): Number of conformers to generate.
        max_attempts (int): Maximum number of attempts.
        rms_thresh (float): RMSD threshold for considering conformers as duplicates.

    Returns:
        list: A list of generated conformers.
    """
    params = AllChem.ETKDG()
    params.numThreads = 0
    params.maxAttempts = max_attempts
    params.pruneRmsThresh = rms_thresh
    conformer_ids = AllChem.EmbedMultipleConfs(molecule, numConfs=num_conformers, params=params)
    results = AllChem.UFFOptimizeMolecule(molecule)

    return conformer_ids, results

def conformer_search(smiles, out_path, num_conformers=1000, max_attempts=10000, rms_thresh=0.2):

    try:
        os.makedirs("{}/conformers".format(out_path))
    except:
        print("Warning, these is already an structures folder in this path, skip mkdir")

    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)  # add H atoms
    
    # conformer generate
    conformer_ids, results = generate_conformers(mol, num_conformers=num_conformers, max_attempts=max_attempts, rms_thresh=rms_thresh)

    # print info
    for i, conf in enumerate(conformer_ids):
        print(f'Conformer {i}:')

        xyz_file = []
        xyz_file.append("{}\n".format(mol.GetNumAtoms()))
        xyz_file.append("conformer_{}\n".format(i))

        for j in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(j)
            symbol = atom.GetSymbol()
            pos = mol.GetConformer(conf).GetAtomPosition(j)
            # print(f'  Atom {j} ({symbol}): x={pos.x:.3f}, y={pos.y:.3f}, z={pos.z:.3f}')

            xyz_file.append("{:6} {:16.8f} {:16.8f} {:16.8f}\n".format(symbol, pos.x, pos.y, pos.z))
            target = open("{}/conformers/conformer_{}.xyz".format(out_path,i), 'w')
            target.writelines(xyz_file)
            target.close()

