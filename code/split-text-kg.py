'''
    Helper script to assist in splitting the text file
    generated from `cook-county-materialization.py`
'''

import os

txt_output_path = "../dataset/cook-county/txt-format-files"
text_kg_file = "cook-county-cases-guilty-verdict.txt"
input_path = os.path.join(txt_output_path, text_kg_file)

file_count = 8 # How many text files do you want?

def split_text_kg(file_count=2):
    with open(input_path, "r") as inp:
        lines = [ line for line in inp.readlines()]

    lines_per_file = int(len(lines)/file_count)
    begin = 0
    end = lines_per_file
    count = 0
    print(f"Total lines to write: {len(lines)}")
    print(f"Approx lines per file: {lines_per_file}")
    for i in range(file_count):
        iter_file_name = ""
        if i < 10:
            iter_file_name = f"cook-county-cases-guilty-verdict-0{count}.txt"
        else:
            iter_file_name = f"cook-county-cases-guilty-verdict-{i}.txt"

        write_path = os.path.join(txt_output_path, iter_file_name)
        with open(write_path, "w") as out:
            for line in lines[begin:end]:
                out.write(line)
        # print(f"{begin}-{end}")
        print(f"Completed iteration: {iter_file_name}")
        begin = end+1
        end += lines_per_file
        count += 1
    

if __name__ == "__main__":
    split_text_kg(file_count)