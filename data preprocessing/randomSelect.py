import random

# Open the input file
with open('train.tsv', 'r', encoding='utf-8') as infile:
    # Read all lines
    lines = infile.readlines()

# Calculate the number of lines to remove, which is 50% of the data
num_lines_to_remove = int(0.50 * len(lines))

# Randomly select the lines to remove
lines_to_remove = random.sample(lines, num_lines_to_remove)

# Open the output file
with open('train_50%.tsv', 'w', encoding='utf-8') as outfile:
    # Write the lines that are not to be removed
    for line in lines:
        if line not in lines_to_remove:
            outfile.write(line)

print("Processing completed. Randomly removed 50% of the data and saved it to the train_50%.txt file.")
