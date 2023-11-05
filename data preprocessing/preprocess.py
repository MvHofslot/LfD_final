# Open the original tsv file and a new file for storing the processed content
with open('test.tsv', 'r', encoding='utf-8') as input_file, open('test_clean.tsv', 'w', encoding='utf-8') as output_file:
    for line in input_file:
        # Remove all occurrences of '@USER' from each line
        cleaned_line = line.replace('@USER', '')
        # Write the processed content to the new file
        output_file.write(cleaned_line)

print("Processing completed. The results have been saved to the processed file.")
