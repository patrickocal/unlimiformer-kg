import json
import sys


if sys.argv[1] == None:
    print("missing commandline argument")
split = str(sys.argv[1])
# File paths
jsonl_file = split + '.jsonl'
replacement_file = 'kg-' + split + '-full-stops.txt'
output_file = 'temp-' + split + '.jsonl'

# Read the replacement lines
with open(replacement_file, 'r') as file:
    replacement_lines = file.readlines()

# Process the JSON Lines file
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

