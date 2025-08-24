element_vdw_radii = {
    # First Period
    'H': 1.20,
    'He': 1.40,
    # Second Period
    'Li': 1.82, 'Be': 1.53, 'B': 1.92, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.35, 'Ne': 1.54,
    # Third Period
    'Na': 2.27, 'Mg': 1.73, 'Al': 1.84, 'Si': 2.10, 'P': 1.80, 'S': 1.80, 'Cl': 1.75, 'Ar': 1.88,
    # Fourth Period
    'K': 2.75, 'Ca': 2.31, 'Sc': 2.11, 'Ti': 1.87, 'V': 1.79, 'Cr': 1.89, 'Mn': 1.97, 'Fe': 1.94, 'Co': 1.92,
    'Ni': 1.63, 'Cu': 1.40, 'Zn': 1.39, 'Ga': 1.87, 'Ge': 2.11, 'As': 1.85, 'Se': 1.90, 'Br': 1.83, 'Kr': 2.02,
    # Fifth Period
    'Rb': None, 'Sr': None, 'Y': None, 'Zr': None, 'Nb': None, 'Mo': None, 'Tc': None, 'Ru': None, 'Rh': None,
    'Pd': None, 'Ag': None, 'Cd': None, 'In': None, 'Sn': None, 'Sb': None, 'Te': 2.06, 'I': 1.98, 'Xe': 2.16}
# comes from https://pubchem.ncbi.nlm.nih.gov/ptable/atomic-radius/

element_covalent_radii = {
    # First Period
    #'H': 0.31,
    'H': 0.31,
    'He': 0.28,
    # Second Period
    'Li': 1.28, 'Be': 0.96, 'B': 0.84, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57, 'Ne': 0.58,
    # Third Period
    'Na': 1.66, 'Mg': 1.41, 'Al': 1.21, 'Si': 1.11, 'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Ar': 1.06,
    # Fourth Period
    'K': 2.03, 'Ca': 1.76, 'Sc': 1.70, 'Ti': 1.60, 'V': 1.53, 'Cr': 1.39, 'Mn': 1.61, 'Fe': 1.52, 'Co': 1.50,
    'Ni': 1.24, 'Cu': 1.32, 'Zn': 1.22, 'Ga': 1.22, 'Ge': 1.20, 'As': 1.19, 'Se': 1.20, 'Br': 1.20, 'Kr': 1.16,
    # Fifth Period
    'Rb': 2.20, 'Sr': 1.95, 'Y': 1.90, 'Zr': 1.75, 'Nb': 1.64, 'Mo': 1.54, 'Tc': 1.47, 'Ru': 1.46, 'Rh': 1.42,
    'Pd': 1.39, 'Ag': 1.45, 'Cd': 1.44, 'In': 1.42, 'Sn': 1.39, 'Sb': 1.39, 'Te': 1.38, 'I': 1.39, 'Xe': 1.40}
# comes from DOI: 10.1039/b801115j

element_masses = {
    # First Period
    'H': 1.0079, 'He': 4.0026,
    # Second Period
    'Li': 6.941, 'Be': 9.0122, 'B': 10.811, 'C': 12.0107, 'N': 14.0067, 'O': 15.9994, 'F': 18.9984, 'Ne': 20.1797,
    # Third Period
    'Na': 22.9897, 'Mg': 24.305, 'Al': 26.9815, 'Si': 28.0855, 'P': 30.9738, 'S': 32.065, 'Cl': 35.453, 'Ar': 39.948,
    # Fourth Period
    'K': 39.0983, 'Ca': 40.078, 'Sc': 44.9559, 'Ti': 47.867, 'V': 50.9415, 'Cr': 51.9961, 'Mn': 54.938, 'Fe': 55.845,
    'Co': 58.9332, 'Ni': 58.6934, 'Cu': 63.546, 'Zn': 65.39, 'Ga': 69.723, 'Ge': 72.64, 'As': 74.9216, 'Se': 78.96,
    'Br': 79.904, 'Kr': 83.8,
    # Fifth Period
    'Rb': 85.4678, 'Sr': 87.62, 'Y': 88.9059, 'Zr': 91.224, 'Nb': 92.9064, 'Mo': 95.94, 'Tc': 98, 'Ru': 101.07,
    'Rh': 102.9055, 'Pd': 106.42, 'Ag': 107.8682, 'Cd': 112.411, 'In': 114.818, 'Sn': 118.71, 'Sb': 121.76, 'Te': 127.6,
    'I': 126.9045, 'Xe': 131.293,
    # Sixth Period
    'Cs': 132.9055, 'Ba': 137.327, 'La': 138.9055, 'Ce': 140.116, 'Pr': 140.9077, 'Nd': 144.24, 'Pm': 145, 'Sm': 150.36,
    'Eu': 151.964, 'Gd': 157.25, 'Tb': 158.9253, 'Dy': 162.5, 'Ho': 164.9303, 'Er': 167.259, 'Tm': 168.9342,
    'Yb': 173.04, 'Lu': 174.967, 'Hf': 178.49, 'Ta': 180.9479, 'W': 183.84, 'Re': 186.207, 'Os': 190.23, 'Ir': 192.217,
    'Pt': 195.078, 'Au': 196.9665, 'Hg': 200.59, 'Tl': 204.3833, 'Pb': 207.2, 'Bi': 208.9804, 'Po': 209, 'At': 210,
    'Rn': 222,
    # Seventh Period
    'Fr': 223, 'Ra': 226, 'Ac': 227, 'Th': 232.0381, 'Pa': 231.0359, 'U': 238.0289, 'Np': 237, 'Pu': 244, 'Am': 243,
    'Cm': 247, 'Bk': 247, 'Cf': 251, 'Es': 252, 'Fm': 257, 'Md': 258, 'No': 259, 'Lr': 262, 'Rf': 261, 'Db': 262,
    'Sg': 266, 'Bh': 264, 'Hs': 277, 'Mt': 268}

periodic_table_list = {
    # First Period
    'H': 1, 'He': 2,
    # Second Period
    'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    # Third Period
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
    # Fourth Period
    'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27,
    'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
    # Fifth Period
    'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45,
    'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54,
    # Sixth Period
    'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63,
    'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72,
    'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81,
    'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86,
    # Seventh Period
    'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95,
    'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104,
    'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109}


def sort_by_atomic_number(all_element_type):
    current_order = []
    for element in all_element_type:
        assert element in periodic_table_list, '{} element is not exist'.format(element)
        current_order.append(periodic_table_list[element])

    sorted_elements = [ELEMENT for ORDER, ELEMENT in sorted(zip(current_order, all_element_type))]

    return sorted_elements

periodic_table_std = {value: key for key, value in periodic_table_list.items()}

# following these website to get help:
# http://img.chem.ucl.ac.uk/sgp/LARGE/sgp.htm
# https://en.wikipedia.org/wiki/Space_group
space_group = {
    1:[["x,y,z"],'P1',"Triclinic"],
    2:[["x,y,z","-x,-y,-z"],'P-1',"Triclinic"],
    4:[["x,y,z","-x,y+1/2,-z"],'P21',"Monoclinic"],
    5:[["x,y,z","-x,y,-z","x+1/2,y+1/2,z","-x+1/2,y+1/2,-z"],'C2',"Monoclinic"],
    7:[["x,y,z","x,-y,1/2+z"],"Pc","Monoclinic"],
    9:[["x,y,z","x,-y,1/2+z","1/2+x,1/2+y,z","1/2+x,1/2-y,1/2+z"],"CC","Monoclinic"],
    12:[["x,y,z","-x,y,-z","1/2+x,1/2+y,z","1/2-x,1/2+y,-z","-x,-y,-z","x,-y,z","1/2-x,1/2-y,-z","1/2+x,1/2-y,z"],"C2/M","Monoclinic"],
    11:[["x,y,z","-x,y+1/2,-z","-x,-y,-z","x,1/2-y,z"],"P21/m","Monoclinic"],
    12:[["x,y,z","x,-y,z","-x,y,-z","-x,-y,-z","1/2+x,1/2+y,z","1/2+x,1/2-y,z","1/2-x,1/2+y,-z","1/2-x,1/2-y,-z"],"C2/m","Monoclinic"],
    13:[["x,y,z","-x,y,1/2-z","-x,-y,-z","x,-y,1/2+z"],"P2/c","Monoclinic"],
    13:[["x,y,z", "-x,y,-z+1/2", "-x,-y,-z", "x,-y,z+1/2"],'P2/c',"Monoclinic"],
    14:[["x,y,z", "-x,y+1/2,-z+1/2", "-x,-y,-z", "x,-y+1/2,z+1/2"],'P21/C',"Monoclinic"],
    15:[["x,y,z","-x,y,1/2-z","1/2+x,1/2+y,z","1/2-x,1/2+y,1/2-z","-x,-y,-z","x,-y,1/2+z","1/2-x,1/2-y,-z","1/2+x,1/2-y,1/2+z"],"C2/C","Monoclinic"],
    18:[["x,y,z","1/2+x,1/2-y,-z","1/2-x,1/2+y,-z","-x,-y,z"],"P21212","Orthorhombic"],
    19:[["x,y,z","1/2+x,1/2-y,-z","-x,1/2+y,1/2-z","1/2-x,-y,1/2+z"],"P212121","Orthorhombic"],
    29:[["x,y,z","1/2-x,y,1/2+z","1/2+x,-y,z","-x,-y,1/2+z"],"PCA21","Orthorhombic"],
    33:[["x,y,z","1/2-x,1/2+y,1/2+z","1/2+x,1/2-y,z","-x,-y,1/2+z"],"PNA21","Orthorhombic"],
    43:[["x,y,z","1/4-x,1/4+y,1/4+z","1/4+x,1/4-y,1/4+z","-x,-y,z","x,y+1/2,z+1/2","1/4-x,3/4+y,3/4+z","1/4+x,3/4-y,3/4+z","-x,-y+1/2,z+1/2",
         "x+1/2,y,z+1/2","3/4-x,1/4+y,3/4+z","3/4+x,1/4-y,3/4+z","-x+1/2,-y,z+1/2","x+1/2,y+1/2,z","3/4-x,3/4+y,1/4+z","3/4+x,3/4-y,1/4+z","-x+1/2,-y+1/2,z"], "Fdd2", "Orthorhombic"],
    56:[["x,y,z","1/2-x,y,1/2+z","x,1/2-y,1/2+z","1/2+x,1/2+y,-z","-x,-y,-z","1/2+x,-y,1/2-z","-x,1/2+y,1/2-z","1/2-x,1/2-y,z"], "Pccn", "Orthorhombic"],
    60:[["x,y,z","1/2-x,1/2+y,z","x,-y,1/2+z","1/2+x,1/2+y,1/2-z","-x,-y,-z","1/2+x,1/2-y,-z","-x,y,1/2-z","1/2-x,1/2-y,1/2+z"],"Pbcn","Orthorhombic"],
    61:[["x,y,z","1/2-x,1/2+y,z","x,1/2-y,1/2+z","1/2+x,y,1/2-z","-x,-y,-z","1/2+x,1/2-y,-z","-x,1/2+y,1/2-z","1/2-x,-y,1/2+z"],"PBCA","Orthorhombic"],
    62:[["x,y,z","x+1/2,-y+1/2,-z+1/2","-x,y+1/2,-z","-x+1/2,-y,z+1/2","-x,-y,-z","-x+1/2,y+1/2,z+1/2","x,-y+1/2,z","x+1/2,y,-z+1/2"],"Pnma","Orthorhombic"],
    77:[["x,y,z","-x,-y,z","-y,x,1/2+z","y,-x,1/2+z"],"P42","Tetragonal"],
    88:[["x,y,z","-x,-y,z","-y,1/2+x,1/4+z","y,1/2-x,1/4+z","-x,1/2-y,1/4-z","x,1/2+y,1/4-z","y,-x,-z","-y,x,-z",
         "1/2+x,1/2+y,1/2+z","1/2-x,1/2-y,1/2+z","1/2-y,x,3/4+z","1/2+y,-x,3/4+z","1/2-x,-y,3/4-z","1/2+x,y,3/4-z","1/2+y,1/2-x,1/2-z","1/2-y,1/2+x,1/2-z"],"I41/a","Tetragonal"],
    96:[["x,y,z","-x,-y,1/2+z","1/2-y,1/2+x,3/4+z","1/2+y,1/2-x,1/4+z","1/2+x,1/2-y,1/4-z","1/2-x,1/2+y,3/4-z","-y,-x,1/2-z","y,x,-z"],"P43212","Tetragonal"],
    143:[["x,y,z","-y,x-y,z","-x+y,-x,z"],"P3","Hexagonal"],
    147:[["x,y,z","-y,x-y,z","-x+y,-x,z","-x,-y,-z","y,-x+y,-z","x-y,x,-z"],"P-3","Hexagonal"],
    148:[["x,y,z","z,x,y","y,z,x","-x,-y,-z","-z,-x,-y","-y,-z,-x"],"R-3","Trigonal"],
    169:[["x,y,z","-y,x-y,1/3+z","-x+y,-x,2/3+z","-x,-y,1/2+z","x-y,x,1/6+z","y,-x+y,5/6+z"],"P61","Hexagonal"]
}

point_group = {"Triclinic":[["a","b","c"],["alpha","beta","gamma"]],
               "Monoclinic":[["a","b","c"],[90,"beta",90]],
               "Orthorhombic":[["a","b","c"],[90,90,90]],
               "Tetragonal":[["a","a","c"],[90,90,90]],
               "Trigonal":[["a","a","a"],["alpha","alpha","alpha"]],
               "Hexagonal":[["a","a","c"],[90,90,120]],
               "Cubic":[["a","a","a"],[90,90,90]]}

