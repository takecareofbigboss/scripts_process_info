
`cat $1 |grep Iteration|grep loss|awk '{print $6 $9}' > trainloss`
`cat $1 |grep Test\ loss|awk '{print $7}' > testloss`
`sed -i s/,/\ /g trainloss`
`matlab -nodisplay < plotloss.m >testlog`
`rm trainloss testloss testlog`
