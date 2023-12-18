import sys

if sys.argv[1] == None:
    print("command line arguments missing")
split = sys.argv[1]

# File paths
target_file = 'kg-column-' + split + '.txt'
output_file = 'temp-kg-' + split + '-full-stops.txt'

# Read the target lines
with open(target_file, 'r') as file:
    target_lines = file.readlines()


# Process the target 
with open(jsonl_file, 'r') as infile, open(output_file, 'w') as outfile:
    for idx, line in enumerate(infile):
        # Load the JSON object from the line
        json_obj = json.loads(line)

        # Replace the 'input' field with the corresponding line from the replacement file
        # Make sure to strip the newline character from the replacement line
        json_obj['input'] = replacement_lines[idx].strip()

        # Write the modified JSON object back to the new file
        json.dump(json_obj, outfile)
        outfile.write('\n')

