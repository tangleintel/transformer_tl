while getopts 'd:p:i:m:t:b:r:L:' OPTION
do
    case $OPTION in
        d)data_type=$OPTARG
            ;;
        p)pytorch_mode=$OPTARG
            ;;
        i)use_ipex=$OPTARG
            ;;
        m)model=$OPTARG
            ;;
        t)task=$OPTARG
            ;;
        b)batch_size=$OPTARG
            ;;
        r)repeat=$OPTARG
            ;;
        L)logdir=$OPTARG
            ;;
    esac
done

if [ ${data_type} == "bf16" ]; then
    is_bf16="yes"
else
    is_bf16="False"
fi

if [ $pytorch_mode == "jit" ]; then
    is_jit="yes"
else
    is_jit="False"
fi

if [ $use_ipex == "ipex" ]; then
    is_ipex="yes"
else
    is_ipex="False"
fi


script="run_glue.py"
script_args="--model_name_or_path ${model} \
	     --eval_batch_size ${batch_size} \
   	     --task_name ${task} \
	     --bf16 ${is_bf16} \
	     --jit_mode ${is_jit} \
	     --use_ipex ${is_ipex} \
             --do_eval \
             --output_dir ./tmp/"

echo $script_args

repeat=(1)
launcher='python -m intel_extension_for_pytorch.cpu.launch'
log_file_prefix="${use_ipex}_${pytorch_mode}_${data_type}_bs${batch_size}_${task}_repeat${repeat}_throughput"
logdir_throughput="${logdir}/throughput"
launcher_args="--socket_id 0 --enable_jemalloc --log_path ${logdir_throughput} --log_file_prefix ${log_file_prefix}"

run_cmd="$launcher $launcher_args $script $script_args"
eval ${run_cmd}
