echo "Usage: __SCRIPT__ <file-to-be-filtered>"
sed -i '/^%/ d' $1 
