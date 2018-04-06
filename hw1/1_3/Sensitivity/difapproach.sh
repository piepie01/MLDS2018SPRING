rm difapproach_train*
rm difapproach_test*
for i in {1,10,50,100,500,1000,5000,10000}
do
	python3 difapproach.py $i
done
