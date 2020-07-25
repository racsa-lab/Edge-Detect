echo "Usage: __script__ <filename>"
#echo "Usage: __script__ <command> <filename>"
while true; do ps -C "python3 pi_queue.py" -o %cpu,%mem,vsz,rss,cmd >> $1;done;
