import os
import sys
import pandas as pd


target_folder = os.getcwd()


if __name__ == '__main__':
    print(f"Cleaning table in folder: {target_folder}")
    error_folder = os.path.join(target_folder, "error_result")
    if not os.path.exists(error_folder):
        os.makedirs(error_folder)
    cif_folder = os.path.join(target_folder, "cif_result_final")
    csv_file = os.path.join(target_folder, "results_scheduler.csv")

    if not os.path.exists(csv_file):
        print(f"CSV file not found in {target_folder}, skipping...")
        sys.exit(0)
    if not os.path.exists(cif_folder):
        print(f"CIF folder not found in {target_folder}, skipping...")
        sys.exit(0)

    all_crystals = pd.read_csv(csv_file)
    cifs = os.listdir(cif_folder)
    invalid_list = []
    for i, item in all_crystals.iterrows():
        if int(item['stage2_steps']) > 2990 or (item['stage1_steps'] >= 2999 and item["stage2_steps"] == 0) or abs(float(item['stage2_energy'])) > 1e15 or item['file']+".cif" not in cifs:
            invalid_list.append(i)

    if invalid_list:
        invalid_df = all_crystals.iloc[invalid_list]
        all_crystals.drop(index=invalid_list, inplace=True)

    all_crystals.sort_values(by=["stage2_energy"], ascending=True, inplace=True)
    all_crystals.reset_index(drop=True, inplace=True)

    min_energy = all_crystals['stage2_energy'][0]
    all_crystals['relative_energy'] = all_crystals['stage2_energy'] - min_energy

    for cif in cifs:
        if cif[:-4] not in all_crystals['file'].values:
            # move cif to error folder
            src_path = os.path.join(cif_folder, cif)
            dest_path = os.path.join(error_folder, cif)
            os.rename(src_path, dest_path)
            
    # Save the cleaned DataFrame back to CSV
    all_crystals.to_csv(csv_file, index=False)