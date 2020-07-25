#!/bin/sh
echo "Usage: script <filename>" 
top -b -n 4000 -d 1 | grep -E "KiB|pi_queue.py" >> $1

