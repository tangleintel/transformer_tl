while getopts 't:L:p:s:m:t:d:r:M:b:B:' OPTION
do
    case $OPTION in
        t)task_type=$OPTARG
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

cd ./${task_type}

./run_batch_model_throughput_mode.sh
./run_batch_model_latency_mode.sh
