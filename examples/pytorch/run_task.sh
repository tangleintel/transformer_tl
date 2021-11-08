while getopts 't:m:' OPTION
do
    case $OPTION in
	t)task_type=$OPTARG
	    ;;
	m)mode=$OPTARG
	    ;;
    esac
done

case $task_type in
    text-classification)
	echo "text-classification"
	./${task_type}/run_batch_models_${task_type}.sh -m $mode 
	;;
    token-classification)
	echo "token-classification"
	./${task_type}/run_batch_models_${task_type}.sh -m $mode 
	;;
    translation)
	echo "translation"
	./${task_type}/run_batch_models_${task_type}.sh -m $mode 
	;;
    text-generation)
	echo "text-generation"
	;;
    question-answering)
	echo "question-answering"
	;;
    summarization)
	echo "summarization"
	;;
esac
