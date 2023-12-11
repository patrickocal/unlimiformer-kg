# Read content of file2.txt into a variable (with newlines replaced by a special token)
#new=$(cat "$3" | tr '\n' '\f')
#old=$(echo "$2" | sed 's/[][\/.^$*]/\\&/g')
#new="$(awk '{printf "%s<newline>", $0}' "$3")"
new=$(cat "$3" | sed 's/\\/\\\\/g' | awk '{printf "%s<newline>", $0}')
#new=$(printf '%s' "$(cat "$3")")
old="$2"
echo "$old"
echo "$new"

# Use awk to replace a line in file1.txt and revert the token back to newlines
awk -v old="$old" -v new="$new" '{
    if ($0 ~ old) 
      {gsub("<newline>", "\n", new); print new}
    else 
      {print $0}
}' $1 > temp-main.tex && mv temp-main.tex $1
