#Shuffle : 1-3-1

	cd Shuffle
	execute shuffle.py to get result
	result would be written in the following files with format "value"
		./shuffle_trainacc
		./shuffle_trainloss
		./shuffle_testacc
		./shuffle_testloss

#Param : 1-3-2

	cd Param
	execute difparam.sh to get result
	result would be written in the following files with format "param value"
		./difparam_trainloss
		./difparam_testloss
		./difparam_trainacc
		./difparam_testacc

#Interpolation : 1-3-3-1

	cd Interpolation
	execute interpolar.py to get result
	result would be written in the following files with format "alpha value"
		./interpolar_trainloss
		./interpolar_testloss
		./interpolar_trainacc
		./interpolar_testacc

#Sensitivity : 1-3-3-2	

	cd Sensitivity
	execute difapproach.sh to get result
	result would be written in the following files with format "batch_size value"
		./difapproach_trainloss
		./difapproach_testloss
		./difapproach_trainacc
		./difapproach_testacc
		./difapproach_sensitivity
