import os
from runner_al import structure 
from runner_al import active_learning as al

data_dir = "data"
al_log_file = "AL.log"
runner_executable = "RuNNer.x"
add_structures = 20
committee_size = 2
total_iterations = 2
num_threads = "2"

input_data = os.path.join(data_dir, "input.data")
pickle_file = os.path.join(data_dir, "structures.pkl")

structure.pickle_file = pickle_file
structure.input_data = input_data
al.pickle_file = pickle_file
al.runner_executable = runner_executable
al.add_structures = add_structures
al.committee_size = committee_size 
al.total_iterations = total_iterations 
al.al_log_file = al_log_file
al.num_threads = num_threads

if __name__ == "__main__":
    structure.main()
    al.main()
