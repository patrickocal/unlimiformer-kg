import json
import sys

if sys.argv[1] == None:
    print("target filename missing")
pathtofile=sys.argv[1]
k = 1
with open(pathtofile, 'r') as f:
    while True:
        line = f.readline()
        if not line:
            break
        try:
            json.loads(line)
        except json.JSONDecodeError as e:
            error_position = e.pos  # Position where the error was detected
            print("line", k, f"JSON is invalid: {e}")
            print(f"Error around character {error_position}:")
            print(line[max(0, error_position - 5):error_position + 5])  # Adjust numbers for broader context
        k += 1
#    try:
#        data = json.load(f)
#        print("JSON is valid!")
#    except json.JSONDecodeError as e:
#        print("JSON is invalid:", e)

