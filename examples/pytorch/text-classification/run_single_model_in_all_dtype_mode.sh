#./run_single_model_in_throughput_mode_text-classification.sh -d fp32 -p jit -i ipex -m bert-base-cased -t MNLI -b 160 -L log
#./run_single_model_in_latency_mode_text-classification.sh -d fp32 -p jit -i stpt -m bert-base-cased -t MNLI -b 160 
while getopts 'p:' OPTION
do
    case $OPTION in
        p)platform=$OPTARG
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

for bs in ${batch_sizes[@]}; do
    for datatype in ${precision[@]}; do
        for ipex in ${use_ipex[@]}; do
	    for((idx=0;idx<${repeat};idx++)); do
	        for pytorch_mode_i in ${pytorch_mode[@]}; do
		    ./run_single_model_in_throughput_mode_text-classification.sh -d $datatype -i $ipex -p $pytorch_mode_i -m bert-base-cased -r idx -t MNLI -b $bs -L log/bert-base-cased-new
		    sleep 30
		done
	    done
	done
    done
done
