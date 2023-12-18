#!/bin/bash
# in this script we combine the files in the folder into one file consisting of
# a single column

process_files() {
  local j=$1
  local min_index=$2
  local max_index=$3
  local inputdir=$4
  local inputfile=$5
  local outputfile=$6

  echo "processing job ID ${j}"
  for ((i = min_index; i <= max_index; i++)); do
    #echo "processing task ID ${i}"
    local kg_file="${inputdir}/generate_triples_${j}_${i}.out"
    if [ ! -e "${kg_file}" ] || [ ! -s "${kg_file}" ]; then
        echo "${kg_file} does not exist or is empty"
    else
      local line=$((i + 1))
      echo "processing task ID ${i}"
      echo "processing line ${line}"
      local kg_str=$(cat "${kg_file}")
      awk -v lineno=$line -v appendStr="$kg_str" 'NR == lineno {sub(/\}$/, appendStr "}"); print; next} {print}' "$inputfile" > "$outputfile"
    fi
  done
}

index=0
inputdir="output-train"
inputfile="train.jsonl"
outputfile="temp-train.jsonl"
JID=(6706295 6613858 6613870 6613874 6613876 6613911 6646044 6646068 6681839 6682018 6682137 6687129 6687140 6688127 6692281 6692285 6693989)
j=6706295

# do the first element separately (since only this has a zero element)
process_files ${j}  0 0 ${inputdir} ${inputfile} ${outputfile}
((index++))
echo ${index}

 iterate across all the other batches
for j in "${JID[@]}"; do
  process_files ${j} 1 999 ${inputdir} ${outputfile}
  ((index++))
done
echo ${index}

# the final batch has only 473
j=6725703
echo "processing job ID ${j}"
process_files ${j} 1 473 ${inputdir} ${outputfile}

((index++))
echo ${index}






# the original loop once more
#for i in {0..0}; do
#  #echo "processing task ID ${i}"
#  kg_file="${inputdir}/generate_triples_${j}_${i}.out"
#  if [ ! -e "${kg_file}" ] || [ ! -s "${kg_file}" ]; then
#    echo "${kg_file} does not exist or is empty"
#  else
#    echo "processing task ID ${i}"
#    kg_str=$(cat "${kg_file}")
#    #awk -v lineno=$i -v appendStr="$kg_str" 'NR == lineno {print $0 appendStr; next} {print}' "$outputfile" > temp && mv temp "$outputfile"
#    #cat ${inputdir}/generate_triples_${j}_${i}.out >> kg-column-train.txt
#    sed -i.bak "${i} s|\}$|${kg_str}\}|" ${outputfile}
#  fi
#done
