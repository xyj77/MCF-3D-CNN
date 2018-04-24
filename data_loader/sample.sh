#!/bin/bash
echo "from sh file "
echo 'start call py'

for file in $(ls)
do
    if [ -d $file ];then
        echo $file
	    python SampleData.py -s 0 ${file}"/train.txt" 128 ${file}"/trainSample.txt"
		python SampleData.py -s 0 ${file}"/test.txt" 64 ${file}"/testSample.txt"
    fi
done

echo 'end call py'