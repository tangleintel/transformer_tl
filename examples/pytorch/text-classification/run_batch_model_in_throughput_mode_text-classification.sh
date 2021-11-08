while getopts 'p:' OPTION
do
    case $OPTION in
        p)platform=$OPTARG
	    ;;
    esac
done


models=`awk '$1 != "#" {print $1}' models_text_classification_config.txt`
model_array=()
for model in $models
do
    model_array[${#model_array[@]}]=$model
done

tasks=`awk '$1 != "#" {print $2}' models_text_classification_config.txt`
task_array=()
for task in $tasks
do
    task_array[${#task_array[@]}]=$task
done

back_archs=`awk '$1 != "#" {print $3}' models_text_classification_config.txt`
back_archs_array=()
for back_arch in $back_archs
do
    back_archs_array[${#back_archs_array[@]}]=$back_arch
done

for((i=0;i<${#model_array[@]};i++))
do
    echo ${model_array[i]}  ${task_array[i]}
    ./run_single_model_in_all_dtype_mode_in_throughput_mode.sh -m ${model_array[i]} -t ${task_array[i]} -p ${platform}
done
