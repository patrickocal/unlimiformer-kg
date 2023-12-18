#!/bin/bash
# substitute everything after /, "input": / on each line of *.jsonl with text from kg-column-*.txt
#awk 'NR==FNR {lines[NR] = $0; next} {sub(/\"input\": ".*?\", \"output\": null}/, "\", \"input\": \"" lines[FNR] "\", \"output\": null}"); print}' \
#kg-column-test.txt test.jsonl > temp-test.jsonl
#awk 'NR==FNR {lines[NR] = $0; next} {sub(/\", \"input\": .*?\", \"output\": \"/, "\", \"input\": \"" lines[FNR] "\", \"output\": \""); print}' \
#kg-column-val.txt validation.jsonl > temp-dev.jsonl
#awk 'NR==FNR {lines[NR] = $0; next} {gsub(/, \"input": /, ", \"input\": \"" lines[FNR] "\"}"); print}'
#kg-column-train.txt train.jsonl > temp-train.jsonl
awk 'NR==FNR {lines[NR] = $0; next} {split($0, json, /, \"input\": \".*?\",/); printf "%s, \"input\": \"%s\",%s\n", json[1], lines[FNR], json[2]}' kg-column-test.txt test.jsonl > temp-test.jsonl

awk 'NR==FNR {lines[NR] = $0; next} {split($0, json, /, \"input\": \".*?\",/); printf "%s, \"input\": \"%s\",%s\n", json[1], lines[FNR], json[2]}' kg-column-val.txt validation.jsonl > temp-validation.jsonl

awk 'NR==FNR {lines[NR] = $0; next} {split($0, json, /, \"input\": \".*?\",/); printf "%s, \"input\": \"%s\",%s\n", json[1], lines[FNR], json[2]}' kg-column-train.txt train.jsonl > temp-train.jsonl


