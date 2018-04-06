rm difparam_train*
rm difparam_test*
for i in {1..500}
do
	python3 -u difparam.py $i &
	wait
done
