while getopts 'w:L:p:s:m:t:d:r:M:b:B:' OPTION
do
    case $OPTION in
        w)which_mode=$OPTARG
            ;;
        L)logdir=$OPTARG
            ;;
        p)platform=$OPTARG
            ;;
        s)script=$OPTARG
            ;;
        m)model=$OPTARG
            ;;
        M)mode=$OPTARG
            ;;
        t)task=$OPTARG
            ;;
        d)dataset=$OPTARG
            ;;
        r)repeat=$OPTARG
            ;;
        b)batch_size=$OPTARG
            ;;
        B)back_arch=$OPTARG
            ;;
    esac
done

script="run_glue.py"
script_args="--model_name_or_path ${model} \
	     --batch_size ${batch_size} \
   	     --task_name ${task} \
             --do_eval \
             --output_dir ./tmp/"


launcher='python -m intel_extension_for_pytorch.cpu.launch'
launcher_args="--socket_id 0 --enable_jemalloc"

run_cmd="$launcher $launch_args $script $script_args"
eval ${run_cmd}



function run_throughput_mode_tl(){

    local dtype=$1
    local mode=$2
    local ipex=$3
    local times=$4

    args=""

    if [ ${dtype} == "bf16" ]; then
        args="$args --bf16 yes"
    else
        args="$args --bf16 False"
    fi

    if [ $mode == "jit" ]; then
        args="$args --jit_mode yes"
    else
        args="$args --jit_mode False"
    fi

    if [ $ipex == "cpu" ]; then
        args="$args --use_ipex yes"
    else
        args="$args --use_ipex False"
    fi

    args="$args \
	   --batch_size ${batch_size} \
	   --model_name_or_path ${model} \
   	   --task_name ${task} \
   	   --back_arch ${back_arch} \
           --do_eval \
           --overwrite_output_dir \
           --output_dir ./tmp/"

    echo $args
    #log_file_prefix="${ipex}_${dataset}_${datatype}_${mode}_bs${batch_size}_repeat${times}"
    log_file_prefix="${ipex}_${task}_${datatype}_${mode}_bs${batch_size}_repeat${times}"
    launch_args="$launcher_args --log_file_prefix ${log_file_prefix}"
    run_cmd="$launcher $launch_args $script $args"
    run_cmd_with_tee_log="${run_cmd} 2>&1 | tee ${log_file_prefix}_throughput_mode.log"
    echo "${run_cmd_with_tee_log}"
    #echo "${run_cmd_tl}"
    eval ${run_cmd_with_tee_log}
    #eval ${run_cmd_tl}
}

