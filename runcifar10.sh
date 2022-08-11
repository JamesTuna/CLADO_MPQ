ds='cifar100'
mode=0
for model in  "mobilenetv2_x0_75" "shufflenetv2_x1_5"
do
	echo ${ds}_${model}_${mode}
	python3 FeintLady.py -ds $ds -m $model --ptq_mode $mode --l1 1e-3 --l2 1e-5 --nl1 3 --nl2 50 &> ${ds}_${model}_${mode}.log
done
