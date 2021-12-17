#!/bin/bash
function getApiCount {
	count=$(curl https://api.countapi.xyz/hit/default/65898bf3-085e-4339-b66f-bef550fc9218 | grep -o -P '\d')
}
getApiCount

echo "API Count: $count"
