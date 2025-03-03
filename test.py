# Test if the disagreements are being saved correctly

import pickle
structures_pkl_file = "data/structures.pkl"
with open(structures_pkl_file, "rb") as f:
    print(pickle.load(f))