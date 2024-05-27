for f in *.log; do
	echo ===::: $f :::===
	tr '\r' '\n' < $f | tail -n 1
done
