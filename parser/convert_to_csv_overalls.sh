#!/bin/sh

echo "Usage: __script__ <filename_to_be_converted> <filename.csv>"

cat $1 | grep "KiB" > overall; awk 'NR % 2 == 1 { o=$0 ; next } { print o "," $0 }' overall | awk 'BEGIN { OFS = ","; ORS = "\n" }{print $4,$6,$8, $10,$13,$15,$19}'> $2; rm overall

