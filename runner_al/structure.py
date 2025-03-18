import pickle

def extract_structures(file_path):
    """Reads input file and extracts structures that begin with 'begin' and end with 'end'."""
    structures = []
    current_structure = []
    recording = False
    serial_number = 0

    with open(file_path, "r") as f:
        for line in f:
            stripped_line = line.strip()
            if stripped_line == "begin":
                recording = True
                current_structure = [stripped_line]  # Start a new structure
            elif stripped_line == "end":
                current_structure.append(stripped_line)
                structures.append({
                    "serial_number": serial_number,
                    "structure": "\n".join(current_structure),  # Store full structure
                })
                serial_number += 1
                recording = False
            elif recording:
                current_structure.append(stripped_line)
    
    return {entry["serial_number"]: entry for entry in structures}

def save_structures(data, file_path):
    """Saves extracted structures to a Pickle file."""
    with open(file_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_structures(file_path):
    """Loads structures from a Pickle file."""
    with open(file_path, "rb") as f:
        return pickle.load(f)

def retrieve_structure(serial_number, file_path):
    """Retrieves a structure by its serial number."""
    data = load_structures(file_path)
    return data.get(serial_number, {"message": "Structure not found"})

def main():
    structured_data = extract_structures(input_data)
    save_structures(structured_data, pickle_file)

if __name__ == "__main__":
    main()