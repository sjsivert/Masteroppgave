#!/bin/bash
echo "Experiment title: $1"
echo "Experiment message: $2"

SKIP_COMMIT=$3

export TERM=xterm

function getApiCount {
     count=$(curl https://api.countapi.xyz/hit/default/65898bf3-085e-4339-b66f-bef550fc9218 | grep -o -P '\d+')
}
getApiCount
experiment_title="$1-[$count]"

echo $experiment_title

# Run experiment
python ./src/main.py --experiment "$experiment_title" "$2"

# Skip committing if a third parameter is given to the script
if [[ -z ${SKIP_COMMIT} ]] ; then
	# Add experiment commit
	echo "Committing"
	git add .
	git commit -m "E: $experiment_title: $2"
else
	echo "Skipping commit"
fi
