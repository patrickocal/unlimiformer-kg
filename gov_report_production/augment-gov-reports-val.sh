#!/bin/bash
# in this script we combine the files in the folder into one file consisting of
# a single column
index=0
outputdir="output-val"
outputfile="kg-column-val.txt"
j=6763829
#JID=(6706295 6613858 6613870 6613874 6613876 6613911 6646044 6646068 6681839 6682018 6682137 6687129 6687140 6688127 6692281 6692285 6693989)
#j=6706295
#echo "processing job ID ${j}"
#for i in {0}; do
#  echo "processing task ID ${i}"
#  cat ${outputdir}/*_${j}_${i}.out >> kg-train-column.txt
#done
#((index++))
#echo ${index}
#for j in "${JID[@]}"; do
#  echo "processing job ID ${j}"
#  # cat the contents of each task into a single file
#  for i in {1..999}; do
#    echo "processing task ID ${i}"
#    cat ${outputdir}/*_${j}_${i}.out >> kg-train-column.txt
#  done
#  ((index++))
#done
echo ${index}
echo "processing job ID ${j}"
for i in {0..971}; do
  sta=$(cat ${outputdir}/*_${j}_${i}.out)
  echo "processing task ID ${i}"
  sed "s|\}$|${sta}\}|" ${outputfile}
done
((index++))
echo ${index}

