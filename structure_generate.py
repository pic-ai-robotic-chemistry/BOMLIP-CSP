from basic_function import format_parser
from basic_function import packaged_function
from basic_function import conformer_search
import time


if __name__ == '__main__':

    time_start = time.time()

    # ##############################################################################################
    # # conformer search
    # conformer_search.conformer_search("C1CC2=COC=C12", "./test/molecule_1", num_conformers=10, max_attempts=10000, rms_thresh=0.1)
    

    # ##############################################################################################

    # ##############################################################################################
    # single crystal structure generate with Z'=1
    
    molecule1 = format_parser.read_xyz_file("./test/molecule_1/conformers/conformer_0.xyz")
    packaged_function.CSP_generater_parallel([molecule1], "./test", need_structure=100, space_group_list=[14,61],add_name="XULDUD_C1", max_workers=16,start_seed=1)
    # ##############################################################################################

    # ##############################################################################################
    # single crystal structure generate with Z'=2
    
    molecule1 = format_parser.read_xyz_file("./test/molecule_1/conformers/conformer_0.xyz")
    packaged_function.CSP_generater_parallel([molecule1,molecule1], "./test", need_structure=100, space_group_list=[14,61],add_name="XULDUD_C1", max_workers=16,start_seed=1)
    # ##############################################################################################

    # ##############################################################################################
    # co-crystal structure generate
    
    molecule1 = format_parser.read_xyz_file("./test/molecule_1/conformers/conformer_0.xyz")
    molecule2 = format_parser.read_xyz_file("./test/molecule_2/conformers/conformer_0.xyz")
    packaged_function.CSP_generater_parallel([molecule1,molecule2], "./test", need_structure=100, space_group_list=[14,61],add_name="XULDUD_C1", max_workers=16,start_seed=1)
    # ##############################################################################################



    time_end=time.time()
    print('time cost',time_end-time_start,'s')
