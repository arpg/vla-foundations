import pickle

# Assuming you have a file named 'data.pkl'
try:
    with open('trajectories.pkl', 'rb') as file:
        data_from_file = pickle.load(file)
        print(f"The data type in the file is: {type(data_from_file)}")
except FileNotFoundError:
    print("File not found.")
except pickle.UnpicklingError:
    print("Error unpickling data. File may be corrupted or not a pickle file.")
except EOFError:
    print("Reached the end of the file.")
