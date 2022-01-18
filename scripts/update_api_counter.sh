NEW_COUNTER_VALUE=${1}

if [[ -z ${NEW_COUNTER_VALUE} ]] ; then
	# Add experiment commit
	echo "Setting counter to 0"
	curl https://api.countapi.xyz/set/default/65898bf3-085e-4339-b66f-bef550fc9218\?value=0
else
	echo "Setting counter to ${NEW_COUNTER_VALUE}"
	curl https://api.countapi.xyz/set/default/65898bf3-085e-4339-b66f-bef550fc9218\?value=${NEW_COUNTER_VALUE}
fi

