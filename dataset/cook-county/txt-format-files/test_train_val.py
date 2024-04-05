import pandas as pd
from sklearn.model_selection import train_test_split

# List of file names
file_names = ['cook-county-cases-guilty-verdict-00.txt', 'cook-county-cases-guilty-verdict-01.txt', 'cook-county-cases-guilty-verdict-02.txt', 'cook-county-cases-guilty-verdict-03.txt', 'cook-county-cases-guilty-verdict-04.txt', 'cook-county-cases-guilty-verdict-05.txt', 'cook-county-cases-guilty-verdict-06.txt', 'cook-county-cases-guilty-verdict-07.txt']

# Loop through each file
for file_name in file_names:
    # Load the dataset
    with open(file_name, 'r') as file:
        data = file.readlines()

    
    # Split the data into training (60%), testing (20%), and validation (20%) sets
    train_data, test_val_data = train_test_split(data, test_size=0.4, random_state=42)
    test_data, val_data = train_test_split(test_val_data, test_size=0.5, random_state=42)

    # Save each set to separate text files
    with open(f'train-{file_name}', 'w') as train_file:
        train_file.writelines(train_data)

    with open(f'test-{file_name}', 'w') as test_file:
        test_file.writelines(test_data)

    with open(f'validate-{file_name}', 'w') as val_file:
        val_file.writelines(val_data)



    print(file_name)