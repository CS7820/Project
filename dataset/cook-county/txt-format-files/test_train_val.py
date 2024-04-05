import pandas as pd
from sklearn.model_selection import train_test_split

# List of file names
file_names = ['cook-county-cases-guilty-verdict-00.csv', 'cook-county-cases-guilty-verdict-01.csv', 'cook-county-cases-guilty-verdict-02.csv', 'cook-county-cases-guilty-verdict-03.csv', 'cook-county-cases-guilty-verdict-04.csv', 'cook-county-cases-guilty-verdict-05.csv', 'cook-county-cases-guilty-verdict-06.csv', 'cook-county-cases-guilty-verdict-07.csv']

# Loop through each file
for file_name in file_names:
    # Load the dataset
    data = pd.read_csv(file_name, dtype=str, low_memory=False)

    
    #Remove columns 4 and 5
    data = data.drop(columns=[data.columns[3], data.columns[4]])
    
    # Split the data into training (60%), testing (20%), and validation (20%) sets
    train_data, test_val_data = train_test_split(data, test_size=0.4, random_state=42)
    test_data, val_data = train_test_split(test_val_data, test_size=0.5, random_state=42)

    # Save each set to separate CSV files
    train_data.to_csv(f'train-{file_name}', index=False)
    test_data.to_csv(f'test-{file_name}', index=False)
    val_data.to_csv(f'validate-{file_name}', index=False)


    print(file_name)