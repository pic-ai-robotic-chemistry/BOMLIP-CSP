import re
from basic_function import operation
from basic_function import data_classes
from basic_function import chemical_knowledge
import copy


def read_xyz_file(file_path):
    input_file = open(file_path, 'r')
    lines = input_file.readlines()
    number_of_atoms = int(lines[0])
    name = str(lines[1][:-1])
    atoms = []
    for index,line in enumerate(lines):
        split_line = list(filter(lambda x: x != '', re.split("\\s+", line)))
        if len(split_line)==4 and operation.is_number(split_line[1]) and \
                operation.is_number(split_line[2]) and operation.is_number(split_line[3]):
            atoms.append(data_classes.Atom(element=split_line[0],
                                            cart_xyz=[float(split_line[1]), float(split_line[2]), float(split_line[3])],
                                           atom_id=index-2))

    if number_of_atoms!=len(atoms):
        print("Warning! The length of atoms don't match the number of atoms given")

    molecule = data_classes.Molecule(atoms=atoms, name=name, system_name=name)

    return molecule

def write_cif_file(crystal, sym=False, name="zcx"):
    """
    Accept crystal class, give the cif file out
    :param crystal: crystal class
    :param coordinates: frac or cart
    :param sym: False:give all atoms out; True:with symmetry
    :param name: file name
    :return: cif_out
    cif file in list format should be print using the following function:
                                        target=open("D:\\zcx.cif",'w')
                                        target.writelines(cif_out)
                                        target.close()
    """
    
    if crystal.system_name!="unknown":
        name = crystal.system_name
    cif_file = []

    cif_file.append("data_"+str(name)+"\n")
    if sym==False:
        if crystal.space_group==1:
            crystal_temp = crystal
        else:
            crystal_temp = copy.deepcopy(crystal)
            crystal_temp.make_p1()
        cif_file.append("_symmetry_space_group_name_H-M    \'P1\'"+"\n")
        cif_file.append("_symmetry_Int_Tables_number       1"+"\n")

        cif_file.append("loop_"+"\n")
        cif_file.append("_symmetry_equiv_pos_site_id"+"\n")
        cif_file.append("_symmetry_equiv_pos_as_xyz"+"\n")
        cif_file.append("1 x,y,z"+"\n")
        cif_file.append("_cell_length_a                   "+str(crystal_temp.cell_para[0][0])+"\n")
        cif_file.append("_cell_length_b                   "+str(crystal_temp.cell_para[0][1])+"\n")
        cif_file.append("_cell_length_c                   "+str(crystal_temp.cell_para[0][2])+"\n")
        cif_file.append("_cell_angle_alpha                "+str(crystal_temp.cell_para[1][0])+"\n")
        cif_file.append("_cell_angle_beta                 "+str(crystal_temp.cell_para[1][1])+"\n")
        cif_file.append("_cell_angle_gamma                "+str(crystal_temp.cell_para[1][2])+"\n")

        cif_file.append("loop_"+"\n")
        cif_file.append("_atom_site_label"+"\n")
        cif_file.append("_atom_site_type_symbol"+"\n")
        cif_file.append("_atom_site_fract_x"+"\n")
        cif_file.append("_atom_site_fract_y"+"\n")
        cif_file.append("_atom_site_fract_z"+"\n")
        for i in range(0,len(crystal_temp.atoms)):
            cif_file.append("{:6}  {:4} {:16.8f} {:16.8f} {:16.8f}\n"
                             .format(i+1,crystal_temp.atoms[i].element,crystal_temp.atoms[i].frac_xyz[0],
                                     crystal_temp.atoms[i].frac_xyz[1],crystal_temp.atoms[i].frac_xyz[2]))

        return cif_file

    elif sym==True:
        cif_file.append("_symmetry_space_group_name_H-M    \'{}\'".format(chemical_knowledge.space_group[crystal.space_group][1])+"\n")
        cif_file.append("_symmetry_Int_Tables_number       {}".format(crystal.space_group)+"\n")

        cif_file.append("loop_"+"\n")
        cif_file.append("_symmetry_equiv_pos_site_id"+"\n")
        cif_file.append("_symmetry_equiv_pos_as_xyz"+"\n")
        for idx, SYMM in enumerate(crystal.SYMM):
            cif_file.append("{} {}".format(idx+1,SYMM)+"\n")
        cif_file.append("_cell_length_a                   "+str(crystal.cell_para[0][0])+"\n")
        cif_file.append("_cell_length_b                   "+str(crystal.cell_para[0][1])+"\n")
        cif_file.append("_cell_length_c                   "+str(crystal.cell_para[0][2])+"\n")
        cif_file.append("_cell_angle_alpha                "+str(crystal.cell_para[1][0])+"\n")
        cif_file.append("_cell_angle_beta                 "+str(crystal.cell_para[1][1])+"\n")
        cif_file.append("_cell_angle_gamma                "+str(crystal.cell_para[1][2])+"\n")

        cif_file.append("loop_"+"\n")
        cif_file.append("_atom_site_label"+"\n")
        cif_file.append("_atom_site_type_symbol"+"\n")
        cif_file.append("_atom_site_fract_x"+"\n")
        cif_file.append("_atom_site_fract_y"+"\n")
        cif_file.append("_atom_site_fract_z"+"\n")
        for i in range(0,len(crystal.atoms)):
            cif_file.append("{:6}  {:4} {:16.8f} {:16.8f} {:16.8f}\n"
                             .format(i+1,crystal.atoms[i].element,crystal.atoms[i].frac_xyz[0],
                                     crystal.atoms[i].frac_xyz[1],crystal.atoms[i].frac_xyz[2]))

        return cif_file


def write_cifs_file(crystals, sym=False, name="zcx"):
    cifs_file = []
    for crystal in crystals:
        single_cif = write_cif_file(crystal,sym=sym, name=name)
        cifs_file.extend(single_cif)
    return cifs_file


def read_cif_file(file_path,on_sym_check=False,shut_up=False,system_name="unknown",comment_name="unknown"):
    input_file = open(file_path, 'r')
    lines = input_file.readlines()
    step_pickle = []
    crystal_all = []
    if system_name=="unknown":
        no_name = True
    else:
        no_name = False
    # first time scan
    for index,line in enumerate(lines):
        # find out all the step pickle
        if line.startswith("data_"):
            step_pickle.append(index)
    step_pickle.append(len(lines))

    # treat every step and return a crystal
    for m in range(0,len(step_pickle)-1):
        atoms = []
        atoms_P1 = []
        SYMM = []
        cell_para = [["unknown","unknown","unknown"],["unknown","unknown","unknown"]]

        for index, line in enumerate(lines[step_pickle[m]:step_pickle[m+1]]):
            split_line = list(filter(lambda x: x != '', re.split("\\s+", line)))
            if line.startswith("#"):
                pass
            elif len(split_line)==0:
                pass
            # read the loop of symmetry
            # elif split_line[0]=="loop_" and lines[step_pickle[m]+index+1]=="_symmetry_equiv_pos_as_xyz\n":
            elif split_line[0] == "loop_" and lines[step_pickle[m] + index + 1] == "_symmetry_equiv_pos_as_xyz\n":
                temp_number = 1
                while "_" not in lines[step_pickle[m]+index+1+temp_number]:
                    split_line_temp = list(filter(lambda x: x != '', re.split("\\s+", lines[step_pickle[m]+index+1+temp_number])))
                    temp_number+=1
                    if not operation.is_number(split_line_temp[0]):
                        SYMM.append(split_line_temp[0])
                    else:
                        SYMM.append(split_line_temp[1])
            elif split_line[0] == "loop_" and lines[step_pickle[m]+index+2].strip(" ")=="_symmetry_equiv_pos_as_xyz\n":
                temp_number = 1
                while "_" not in lines[step_pickle[m]+index+2+temp_number]:
                    split_line_temp = list(filter(lambda x: x != '', re.split("\\s+", lines[step_pickle[m]+index+2+temp_number])))
                    temp_number+=1
                    SYMM.append("".join(split_line_temp[1:]))
            elif split_line[0] == "loop_" and "_space_group_symop_operation_xyz\n" in lines[step_pickle[m] + index + 1]:
                # ase format
                temp_number = 1
                while "_" not in lines[step_pickle[m]+index+1+temp_number]:
                    if lines[step_pickle[m]+index+1+temp_number]=="\n":
                        temp_number += 1
                        continue
                    split_line_temp = list(filter(lambda x: x != '', re.split("\\s+", lines[step_pickle[m]+index+1+temp_number])))
                    temp_number+=1
                    if not operation.is_number(split_line_temp[0]):
                        SYMM.append("".join(split_line_temp))

            # read the loop of atoms:
            elif (split_line[0] == "loop_" and lines[step_pickle[m] + index + 1].strip(" ") == "_atom_site_label\n") or \
                    (split_line[0] == "loop_" and lines[step_pickle[m] + index + 2].strip(" ") == "_atom_site_label\n"):
                temp_number = 0
                while "_" in lines[step_pickle[m]+index+1+temp_number]:
                    if lines[step_pickle[m] + index + 1 + temp_number].strip(" ") == "_atom_site_type_symbol\n":
                        ele_pos = temp_number
                    elif lines[step_pickle[m] + index + 1 + temp_number].strip(" ") == "_atom_site_fract_x\n":
                        x_pos = temp_number
                    elif lines[step_pickle[m] + index + 1 + temp_number].strip(" ") == "_atom_site_fract_y\n":
                        y_pos = temp_number
                    elif lines[step_pickle[m] + index + 1 + temp_number].strip(" ") == "_atom_site_fract_z\n":
                        z_pos = temp_number
                    temp_number+=1
                how_long = temp_number

                while len(list(filter(lambda x: x != '', re.split("\\s+", lines[step_pickle[m]+index+1+temp_number])))) == how_long:
                    split_line_temp = list(filter(lambda x: x != '', re.split("\\s+", lines[step_pickle[m]+index+1+temp_number])))
                    atoms.append(data_classes.Atom(element=split_line_temp[ele_pos],
                                           frac_xyz=[float(split_line_temp[x_pos]),float(split_line_temp[y_pos]),
                                                     float(split_line_temp[z_pos])]))
                    temp_number += 1
                    if step_pickle[m]+index+1+temp_number==len(lines):
                        break

            elif split_line[0] == "_cell_length_a":
                cell_para[0][0] = float(split_line[1])
            elif split_line[0] == "_cell_length_b":
                cell_para[0][1] = float(split_line[1])
            elif split_line[0] == "_cell_length_c":
                cell_para[0][2] = float(split_line[1])
            elif split_line[0] == "_cell_angle_alpha":
                cell_para[1][0] = float(split_line[1])
            elif split_line[0] == "_cell_angle_beta":
                cell_para[1][1] = float(split_line[1])
            elif split_line[0] == "_cell_angle_gamma":
                cell_para[1][2] = float(split_line[1])
            elif "data_" in line:
                if no_name == True:
                    system_name = line[5:]
                    system_name = system_name.replace(" ","_")
                    system_name = system_name.replace("\\", "_")
                    system_name = system_name.replace("\n", "")
        for atom in atoms:
            all_reflect_position = operation.space_group_transfer_for_single_atom(atom.frac_xyz, SYMM)
            for new_position in all_reflect_position:
                atoms_P1.append(data_classes.Atom(element=atom.element,
                                            frac_xyz=[new_position[0], new_position[1], new_position[2]]))
        crystal_all.append(data_classes.Crystal(cell_para=cell_para, atoms=atoms_P1, comment=comment_name, system_name=system_name))
        if on_sym_check == True:
            raise Exception("Not finished part, TODO in code")
        if shut_up==False:
            if m%100 == 0:
                print("{} structures have been treated".format(m))

    return crystal_all


def write_poscar_file(crystal, coordinates = 'frac', name = "parser_zcx_create"):

    vasp_file = []

    vasp_file.append('{}\n'.format(name))
    vasp_file.append('1.0\n')
    cell_vect = crystal.cell_vect
    for vect in cell_vect:
        vasp_file.append("{:16.8f} {:16.8f} {:16.8f}\n".format(vect[0],vect[1],vect[2]))
    crystal.sort_by_element()
    vasp_file.append("".join("{:>6s}".format(x) for x in crystal.get_element()) + "\n")
    vasp_file.append("".join("{:>6.0f}".format(x) for x in crystal.get_element_amount()) + "\n")
    if coordinates == 'frac':
        vasp_file.append('Direct\n')
        for ELEMENT in crystal.get_element():
            for ATOM in crystal.atoms:
                if ATOM.element == ELEMENT:
                    vasp_file.append(
                        "{:16.8f} {:16.8f} {:16.8f}\n".format(ATOM.frac_xyz[0], ATOM.frac_xyz[1], ATOM.frac_xyz[2]))
    elif coordinates == 'cart':
        vasp_file.append('Cartesian\n')
        for ELEMENT in crystal.get_element():
            for ATOM in crystal.atoms:
                if ATOM.element == ELEMENT:
                    vasp_file.append(
                        "{:16.8f} {:16.8f} {:16.8f}\n".format(ATOM.cart_xyz[0], ATOM.cart_xyz[1], ATOM.cart_xyz[2]))
    else:
        raise Exception("Wrong coordinates type: {}".format(coordinates))

    return vasp_file
    
    
def read_ase_pbc_file(file_path,shut_up=False):
    input_file = open(file_path, 'r')
    lines = input_file.readlines()[2:]
    step_pickle = []
    crystal_all = []

    # first time scan
    for index,line in enumerate(lines):
        # find out all the step pickle
        if line.startswith("Step "):
            step_pickle.append(index)
    step_pickle.append(len(lines))

    # treat every step and return a crystal
    for m in range(0,len(step_pickle)-1):
        atoms_P1 = []
        force_matrix = []
        position_matrix = []
        in_forces = False
        in_positions = False


        for index, line in enumerate(lines[step_pickle[m]:step_pickle[m+1]]):
            # split_line = list(filter(lambda x: x != '', re.split("\\s+", line)))
            line = line.strip()
            # check Forces part
            if line.startswith("Forces:"):
                in_forces = True
                in_positions = False
                continue

            # check Positions part
            if line.startswith("Positions:"):
                in_positions = True
                in_forces = False
                continue

            if in_forces and line.startswith("[") and line.endswith("]"):
                line = line.replace("[", "").replace("]", "")
                force_matrix.append([float(x) for x in line.split()])

            # analyse Positions part
            if in_positions and line.startswith("[") and line.endswith("]"):
                line = line.replace("[", "").replace("]", "")
                position_matrix.append([float(x) for x in line.split()])

            if line.startswith("Elements:"):
                elements_string = line.strip().split(":", 1)[-1].strip()
                elements_string = elements_string[1:-1]  

                elements = [elem.strip().strip("'") for elem in elements_string.split(",")]

            if line.startswith("cell:"):
                matrix_string = line[len("cell: Cell("):-1]
                rows = matrix_string.split("], [")
                cell_vect = [
                    [float(value) for value in row.replace('[', '').replace(']', '').replace(')', '').split(", ")]
                    for row in rows
                ]

        for i in range(0,len(elements)):
            atoms_P1.append(data_classes.Atom(element=elements[i], cart_xyz=[position_matrix[i][0], position_matrix[i][1], position_matrix[i][2]]))

        crystal_all.append(data_classes.Crystal(cell_vect=cell_vect, atoms=atoms_P1))

    return crystal_all    
    