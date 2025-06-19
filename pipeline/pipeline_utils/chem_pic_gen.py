import os
from rdkit import Chem
from rdkit.Chem import Draw
import requests

def generate_molecule_diagram(name_or_smiles, out_dir="molecule_images", use_pubchem=False):
    os.makedirs(out_dir, exist_ok=True)
    filename = f"{out_dir}/{name_or_smiles.replace(' ', '_')}.png"

    if not use_pubchem:
        try:
            # Try interpreting input as SMILES first
            mol = Chem.MolFromSmiles(name_or_smiles)
            if mol is None:
                # Try resolving molecule name to SMILES using PubChem
                r = requests.get(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name_or_smiles}/property/IsomericSMILES/JSON")
                smiles = r.json()['PropertyTable']['Properties'][0]['IsomericSMILES']
                mol = Chem.MolFromSmiles(smiles)
            Draw.MolToFile(mol, filename)
            return filename
        except Exception as e:
            print(f"RDKit error: {e}")

    # Fall back to PubChem if RDKit fails or use_pubchem=True
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name_or_smiles}/PNG"
        r = requests.get(url)
        with open(filename, 'wb') as f:
            f.write(r.content)
        print(f"Generated molecule diagram for {name_or_smiles} and saved to {filename}")
    except Exception as e:
        print(f"PubChem error: {e}")
        return None

def main():
    """Main method to test molecule diagram generation"""
    
    # Test compounds - using both common names and SMILES
    test_compounds = [
        "aspirin",
        "caffeine", 
        "glucose",
        "CCO",  # ethanol SMILES
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # aspirin SMILES
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"  # caffeine SMILES
    ]
    
    print("Testing molecule diagram generation...")
    print("=" * 50)
    
    for compound in test_compounds:
        print(f"Generating diagram for: {compound}")
        try:
            result = generate_molecule_diagram(compound)
            if result:
                print(f"✓ Successfully generated: {result}")
            else:
                print(f"✗ Failed to generate diagram for: {compound}")
        except Exception as e:
            print(f"✗ Error generating diagram for {compound}: {e}")
        print("-" * 30)
    
    print("Test completed!")

if __name__ == "__main__":
    main()
