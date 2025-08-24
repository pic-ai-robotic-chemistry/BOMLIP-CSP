from basic_function import format_parser
from basic_function import CSP_generator_normal
import os
import concurrent.futures
import sys



def process_crystal(seed, sg, molecules,output_path,add_name):
    aaa = CSP_generator_normal.CrystalGenerator(molecules, space_group=sg)
    molecules_number = sum(len(molecule.atoms) for molecule in molecules)
    new_crystal = aaa.generate(seed=seed)
    sys.stdout.flush()
    if new_crystal is not None:
        cif_out = format_parser.write_cif_file(new_crystal)
        with open(f"{output_path}/structures/{add_name}_{sg}_{seed}_z{len(molecules)}_{molecules_number}.cif", 'w') as target:
            target.writelines(cif_out)
        return True
    return False

def CSP_generater_parallel(molecules,output_path,need_structure = 100, space_group_list=[1],max_workers=8,add_name='',start_seed=1):
    space_groups = space_group_list
    accept_count = need_structure

    try:
        os.makedirs("{}/structures".format(output_path))
    except:
        print("Warning, these is already an structures folder in this path, skip mkdir")
    for sg in space_groups:
        accept = 0
        seed = start_seed

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            while accept < accept_count:
                # submit new task
                while len(futures) < max_workers and accept + len(futures) < accept_count:
                    future = executor.submit(process_crystal, seed, sg, molecules, output_path,add_name)
                    futures[future] = seed
                    seed += 1

                # check the finished task
                done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    if future.result():
                        accept += 1
                    # remove it from list, no matter what result it is
                    del futures[future]

                # cancel all task if the number need is arrived.
                if accept >= accept_count:
                    for future in futures:
                        future.cancel()
                    break


def CSP_generater_serial(molecules,output_path,need_structure = 100, densely_pack_method=False, space_group_list=[1]):
    """
    :param molecules: a list [molecule1, molecule2, ...]
    :param output_path: a str indicate the path of output folder
    :param need_structure: int
    :param space_group_list:a list indicate the space group need to search
    """
    try:
        os.makedirs("{}\\structures".format(output_path))
    except:
        print("Warning, these is already an structures folder in this path, skip mkdir")
    for sg in space_group_list:
        aaa = CSP_generator_normal.CrystalGenerator(molecules, space_group=sg)
        accept=0
        i=1
        while accept<need_structure:
            new_crystal = aaa.generate(seed=i,densely_pack_method=densely_pack_method)
            if new_crystal==None:
                i += 1
                continue
            cif_out = format_parser.write_cif_file(new_crystal)
            target = open("{}\\structures\\{}_{}.cif".format(output_path,sg,i), 'w')
            target.writelines(cif_out)
            target.close()
            accept+=1
            i += 1


def CSP_generater_one_test(molecules,output_path,space_group=1,seed=1):
    """
    used to test the generator for a given space group and seed
    :param molecules: a list [molecule1, molecule2, ...]
    :param output_path: a str indicate the path of output folder
    :param space_group: a list indicate the space group need to search
    :param seed: int
    :return: write out a cif
    """
    aaa = CSP_generator_normal.CrystalGenerator(molecules, space_group=space_group)
    new_crystal = aaa.generate(seed=seed, test=True)
    if new_crystal==None:
        print("Return failed generate")
    else:
        cif_out = format_parser.write_cif_file(new_crystal,sym=True)
        target = open("{}\\{}_{}.cif".format(output_path,space_group,seed), 'w')
        target.writelines(cif_out)
        target.close()
