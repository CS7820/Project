import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Directory containing the text files
data_directory = 'CS7820/Project/tree/main/dataset/cook-county/txt-format-files'

# Get a list of all text files in the directory
file_names = os.listdir(data_directory)

# Iterate over each file
for file_name in file_names:
    if file_name.endswith('.txt'):
        # Initialize an empty list to store valid data
        valid_data = []
        # Open the file and read line by line
        with open(os.path.join(data_directory, file_name), 'r') as file:
            for line in file:
                # Split the line by tab delimiter
                fields = line.strip().split('\t')
                # Check if the line has exactly 3 fields
                if len(fields) == 3:
                    # If so, add it to the list of valid data
                    valid_data.append(fields)
        # Convert the list of valid data to a DataFrame
        data = pd.DataFrame(valid_data)

    # Split the data into training (60%), testing (20%), and validation (20%) sets
    train_data, test_val_data = train_test_split(data, test_size=0.4, random_state=42)
    test_data, val_data = train_test_split(test_val_data, test_size=0.5, random_state=42)
        
    # Save each set to separate TSV files in the same directory
    train_data.to_csv(os.path.join(data_directory, f'train-{file_name[:-4]}.tsv'), sep='\t', index=False)
    test_data.to_csv(os.path.join(data_directory, f'test-{file_name[:-4]}.tsv'), sep='\t', index=False)
    val_data.to_csv(os.path.join(data_directory, f'validate{file_name[:-4]}.tsv'), sep='\t', index=False)
	
    print(file_name)