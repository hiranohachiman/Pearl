ckpt=`find . -name "*.ckpt" | xargs ls -lt | awk '{ print $NF }' | grep -v '^$' | head -n 2 | tail -n 1`
echo $ckpt
poetry run python validate/validate.py --model=$ckpt --pearl --coef