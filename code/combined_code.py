import os 

# Directory containing the text files
data_directory = '../dataset/cook-county/txt-format-files'

file_names = ["validatecook-county-cases-guilty-verdict-00.txt", "validatecook-county-cases-guilty-verdict-01.txt", "validatecook-county-cases-guilty-verdict-02.txt", "validatecook-county-cases-guilty-verdict-03.txt", "validatecook-county-cases-guilty-verdict-04.txt", "validatecook-county-cases-guilty-verdict-05.txt", "validatecook-county-cases-guilty-verdict-06.txt", "validatecook-county-cases-guilty-verdict-07.txt"]
output_file = "validate_combined.txt"

with open(output_file, "w") as outfile:
    for file_name in file_names:
        with open(os.path.join(data_directory, file_name), "r") as infile:
            outfile.write(infile.read() + "\n")