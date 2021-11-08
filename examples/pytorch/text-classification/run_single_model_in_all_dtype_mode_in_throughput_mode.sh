while getopts 'p:m:t:' OPTION
do
    case $OPTION in
        p)platform=$OPTARG
	    ;;
        m)model=$OPTARG
	    ;;
        t)task=$OPTARG
	    ;;
    esac
done

batch_sizes=(1)
CORES=`lscpu | grep Core | awk '{print $4}'`
bs_large=`expr 4 \* $CORES`
batch_sizes+=($bs_large)

precision=("fp32")
if [ "$platform" == "CPX" ]; then
    precision=("bf16")
elif [ "$platform" == "SPR" ]; then
    export DNNL_GRAPH_CONSTANT_CACHE=1
    export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
    precision+=("bf16")
elif [ "$platform" == "ICX" ]; then
    precision=("fp32")
fi

use_ipex=("ipex" "stpt")
repeat=(1)
pytorch_mode=("imperative" "jit")

if [[ ! -d log ]];then
    mkdir -p log
fi

logdir="log/${model}"

if [[ ! -d $logdir ]];then
    mkdir -p $logdir
fi



for bs in ${batch_sizes[@]}; do
    for datatype in ${precision[@]}; do
        for ipex in ${use_ipex[@]}; do
	    for((idx=0;idx<${repeat};idx++)); do
	        for pytorch_mode_i in ${pytorch_mode[@]}; do
		    ./run_single_model_in_throughput_mode_text-classification.sh -d $datatype -i $ipex -p $pytorch_mode_i -m $model -r idx -t $task -b $bs -L ${logdir}
		    sleep 25
		done
	    done
	done
    done
done
