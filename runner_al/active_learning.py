import os
import glob
import numpy as np
import pickle
import random
import shutil
import subprocess
from mpi4py import MPI



comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def update_random_seed_in_file(file_path, random_seed_value):
    with open(file_path, "r") as f:
        lines = f.readlines()
    with open(file_path, "w") as f:
        for line in lines:
            if line.startswith("random_seed"):
                f.write(f"random_seed {random_seed_value}\n")
            else:
                f.write(line)

def rename_files(directory, mapping):
    for prefix, new_prefix in mapping.items():
        for filename in os.listdir(directory):
            if filename.startswith(prefix) and filename.endswith(".out"):
                new_filename = filename.replace(prefix, new_prefix).replace(".out", ".data")
                shutil.move(os.path.join(directory, filename), os.path.join(directory, new_filename))

def active_learning():
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = num_threads

    with open(pickle_file, "rb") as f:
        structures_data = pickle.load(f)  # Dictionary with structure IDs
    previous_selected_serials = set()
    for iteration in range(1, total_iterations + 1):
        if iteration == 1:
            all_serials = list(structures_data.keys())
            num_samples = min(add_structures, len(all_serials))
            indices = np.linspace(0, len(all_serials) - 1, num_samples, dtype=int)
            selected_serials = [all_serials[i] for i in indices]
        else:
            std_disagreements = {
                conf_id: data[f"iteration_{iteration -1}"].get("std_disagreement", 0)
                for conf_id, data in structures_data.items()
                if f"iteration_{iteration -1}" in data
            }

            sorted_structures = sorted(std_disagreements, key=std_disagreements.get, reverse=True)
            top_20_structures=[s for s in sorted_structures if s not in previous_selected_serials][:add_structures]
            selected_serials = top_20_structures

        selected_structures = [structures_data[serial]["structure"] for serial in selected_serials]
        previous_selected_serials.update(selected_serials)

        if rank == 0:
            with open(al_log_file, "a") as f:
                f.write(f"\nIteration: {iteration}:\n")
                f.write("\n".join(map(str, selected_serials)))

        selected_serials = comm.bcast(selected_serials, root=0)
        selected_structures = comm.bcast(selected_structures, root=0)

        structure_disagreements = {conf_id: [] for conf_id in structures_data.keys()}

        if rank < committee_size:
            i = rank + 1
            directory = f"iteration_{iteration}/model_{i}"
            os.makedirs(directory, exist_ok=True)

            if iteration == 1:
                with open(f"{directory}/input.data", "a") as f:
                    for serial, structure in zip(selected_serials, selected_structures):
                        f.write(f"{structure}\n")  # Structure separated by newline
            else:
                shutil.copy(f"iteration_{iteration - 1}/model_1/{iteration - 1}_1_input.data", f"{directory}/input.data")
                with open(f"{directory}/input.data", "a") as f:
                    for serial, structure in zip(selected_serials, selected_structures):
                        f.write(f"{structure}\n")  # Structure separated by newline

            random_seed_value = 12345 + iteration + i

            shutil.copy("data/input_1_1.nn", f"{directory}/input.nn")
            update_random_seed_in_file(f"{directory}/input.nn", random_seed_value)

            with open(f"{directory}/mode1_charge.out", "w") as log_file:
                subprocess.run(runner_executable, shell=True, cwd=directory, stdout=log_file, stderr=log_file)

            shutil.copy("data/input_1_2.nn", f"{directory}/input.nn")
            update_random_seed_in_file(f"{directory}/input.nn", random_seed_value)

            with open(f"{directory}/mode2_charge.out", "w") as log_file:
                subprocess.run(runner_executable, shell=True, cwd=directory, stdout=log_file, stderr=log_file)

            rename_files(directory, {"opthardness.": "hardness.", "optweightse.": "weightse."})

            shutil.copy("data/input_2_1.nn", f"{directory}/input.nn")
            update_random_seed_in_file(f"{directory}/input.nn", random_seed_value)

            with open(f"{directory}/mode1_short.out", "w") as log_file:
                subprocess.run(runner_executable, shell=True, cwd=directory, stdout=log_file, stderr=log_file)

            shutil.copy("data/input_2_2.nn", f"{directory}/input.nn")
            update_random_seed_in_file(f"{directory}/input.nn", random_seed_value)

            with open(f"{directory}/mode2_short.out", "w") as log_file:
                subprocess.run(runner_executable, shell=True, cwd=directory, stdout=log_file, stderr=log_file, env=env)

            rename_files(directory, {"optweights.": "weights."})

            shutil.copy("data/input_3.nn", f"{directory}/input.nn")
            update_random_seed_in_file(f"{directory}/input.nn", random_seed_value)

            shutil.move(f"{directory}/input.data", f"{directory}/{iteration}_{i}_input.data")
            shutil.copy("data/input.data", f"{directory}/input.data")
            with open(f"{directory}/mode3.out", "w") as log_file:
                subprocess.run(runner_executable, shell=True, cwd=directory, stdout=log_file, stderr=log_file, env=env)

            nnforces_file = f"{directory}/nnforces.out"

            reference_forces = {}
            prediction_forces = {}

            if os.path.isfile(nnforces_file):
                with open(nnforces_file, "r") as f:
                    for line in f:
                        if line.startswith("Conf.") or line.startswith("#"):
                            continue
                        parts = line.strip().split()
                        if len(parts) == 8:
                            conf_id = int(parts[0]) - 1
                            ref_force = np.array(list(map(float, parts[2:5])))
                            pred_force = np.array(list(map(float, parts[5:8])))

                            if conf_id not in reference_forces:
                                reference_forces[conf_id] = []
                                prediction_forces[conf_id] = []

                            reference_forces[conf_id].append(ref_force)
                            prediction_forces[conf_id].append(pred_force)

                for conf_id in reference_forces:
                    ref_forces = np.array(reference_forces[conf_id])
                    pred_forces = np.array(prediction_forces[conf_id])

                    std_ref = np.std(ref_forces, axis=0)
                    std_pred = np.std(pred_forces, axis=0)

                    rms_ref = np.sqrt(np.mean(std_ref ** 2))
                    rms_pred = np.sqrt(np.mean(std_pred ** 2))

                    disagreement = abs(rms_ref - rms_pred)

                    if conf_id not in structure_disagreements:
                        structure_disagreements[conf_id] = []
                    structure_disagreements[conf_id].append(disagreement)
            else:
                print(f"skipping iteration {iteration}, model {i} as nnforces file is not present")

            file_to_del = [f"{directory}/0*", f"{directory}/output.data" , f"{directory}/input.data", f"{directory}/function*", f"{directory}/nn*", f"{directory}/testing*", f"{directory}/mode3.out"]
            for filename in file_to_del:
                for file in glob.glob(filename):
                    os.remove(file)
        all_structure_disagreements = comm.gather(structure_disagreements, root=0)

        if rank == 0:
            merged_disagreements = {conf_id: [] for conf_id in structures_data.keys()}

            for rank_disagreements in all_structure_disagreements:
                for conf_id, disagreements in rank_disagreements.items():
                    merged_disagreements[conf_id].extend(disagreements)

            for conf_id, disagreements in merged_disagreements.items():
                std_disagreement = np.std(disagreements)

                if conf_id in structures_data:
                    iteration_key = f"iteration_{iteration}"
                    
                    if iteration_key not in structures_data[conf_id]:
                        structures_data[conf_id][iteration_key] = {}

                    structures_data[conf_id][iteration_key]["std_disagreement"] = f"{std_disagreement:.16f}"
                else:
                    print(f"Warning: Structure {conf_id} not found in structures.pkl")

            with open(pickle_file, "wb") as f:
                pickle.dump(structures_data, f)
        comm.Barrier()

def main():
    with open(al_log_file, "w") as f:
        f.write("Active Learning Log\n")
    active_learning()

if __name__ == "__main__":
    main()
