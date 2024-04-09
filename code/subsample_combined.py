import os 
import random 

# Directory containing the text files
data_directory = r"C:\Users\Calvin\Documents\Project\dataset\cook-county\txt-format-files"

input_file = "validate_combined.txt"

# Number of rows to sample
sample_size = 3333

# Path to the output file for sampled rows
output_file = "validate_tiny_sample_combined.txt"

# Read all lines from the input file
with open(input_file, "r") as infile:
    lines = infile.readlines()

# Randomly sample 10,000 lines
sampled_lines = random.sample(lines, sample_size)

# Write sampled lines to the output file
with open(output_file, "w") as outfile:
    for line in sampled_lines:
        outfile.write(line)