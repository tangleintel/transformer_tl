while getopts 'm:B:' OPTION
do
    case $OPTION in
        m)mode=$OPTARG
            ;;
        B)back_arch=$OPTARG
            ;;
    esac
done

case $mode in
    throughput) ./run_batch_model_throughput_mode.sh
	;;
    latency) ./run_batch_model_latency_mode.sh
	;;
esac


#./run_throughput_mode_tl.sh -r 1 -m bert-base-cased -t MNLI -s run_glue.py -p ${platform} -L log/bert-base-cased -b 1 -B bert -w t
#./run_throughput_mode_tl.sh -r 1 -m bert-base-cased -t MNLI -s run_glue.py -p ${platform} -L log/bert-base-cased -b 160 -B bert
#
#./run_throughput_mode_tl.sh -r 1 -m finiteautomata/beto-sentiment-analysis -t MNLI -s run_glue.py -p ICX -L log/finiteautomata/beto-sentiment-analysis -b 1 -B bert
#./run_throughput_mode_tl.sh -r 1 -m finiteautomata/beto-sentiment-analysis -t MNLI -s run_glue.py -p ICX -L log/finiteautomata/beto-sentiment-analysis -b 160 -B bert
#
#./run_throughput_mode_tl.sh -r 1 -m ProsusAI/finbert -t MNLI -s run_glue.py -p ICX -L log/ProsusAI/finbert -b 1 -B bert
#./run_throughput_mode_tl.sh -r 1 -m ProsusAI/finbert -t MNLI -s run_glue.py -p ICX -L log/ProsusAI/finbert -b 160 -B bert
#
#./run_throughput_mode_tl.sh -r 1 -m roberta-large-mnli -t MNLI -s run_glue.py -p ICX -L log/roberta-large-mnli -b 1 -B roberta
#./run_throughput_mode_tl.sh -r 1 -m roberta-large-mnli -t MNLI -s run_glue.py -p ICX -L log/roberta-large-mnli -b 160 -B roberta
#
#./run_throughput_mode_tl.sh -r 1 -m chkla/roberta-argument -t MRPC -s run_glue.py -p ICX -L log/chkla/roberta-argument -b 1 -B roberta
#./run_throughput_mode_tl.sh -r 1 -m chkla/roberta-argument -t MRPC -s run_glue.py -p ICX -L log/chkla/roberta-argument -b 160 -B roberta
#
#./run_throughput_mode_tl.sh -r 1 -m cardiffnlp/twitter-roberta-base-sentiment -t MNLI -s run_glue.py -p ICX -L log/cardiffnlp/twitter-roberta-base-sentiment -b 1 -B roberta
#./run_throughput_mode_tl.sh -r 1 -m cardiffnlp/twitter-roberta-base-sentiment -t MNLI -s run_glue.py -p ICX -L log/cardiffnlp/twitter-roberta-base-sentiment -b 160 -B roberta
#
#./run_throughput_mode_tl.sh -r 1 -m cardiffnlp/twitter-roberta-base-offensive -t MRPC -s run_glue.py -p ICX -L log/cardiffnlp/twitter-roberta-base-offensive -b 1 -B roberta
#./run_throughput_mode_tl.sh -r 1 -m cardiffnlp/twitter-roberta-base-offensive -t MRPC -s run_glue.py -p ICX -L log/cardiffnlp/twitter-roberta-base-offensive -b 160 -B roberta
#
#./run_throughput_mode_tl.sh -r 1 -m typeform/distilbert-base-uncased-mnli -t MNLI -s run_glue.py -p ICX -L log/typeform/distilbert-base-uncased-mnli -b 1 -B distilbert
#./run_throughput_mode_tl.sh -r 1 -m typeform/distilbert-base-uncased-mnli -t MNLI -s run_glue.py -p ICX -L log/typeform/distilbert-base-uncased-mnli -b 160 -B distilbert


#./run_throughput_mode_tl.sh -r 1 -m cardiffnlp/twitter-roberta-base-emotion -t MNLI -s run_glue.py -p ICX -L log/cardiffnlp/twitter-roberta-base-emotion -b 1 -B roberta
#./run_throughput_mode_tl.sh -r 1 -m cardiffnlp/twitter-roberta-base-emotion -t MNLI -s run_glue.py -p ICX -L log/cardiffnlp/twitter-roberta-base-emotion -b 160 -B roberta


#./run_throughput_mode_tl.sh -r 1 -m distilbert-base-uncased-finetuned-sst-2-english -t MNLI -s run_glue.py -p ICX -L log/distilbert-base-uncased-finetuned-sst-2-english -b 1 -B distilbert
#./run_throughput_mode_tl.sh -r 1 -m distilbert-base-uncased-finetuned-sst-2-english -t MNLI -s run_glue.py -p ICX -L log/distilbert-base-uncased-finetuned-sst-2-english -b 160 -B distilbert

#./run_throughput_mode_tl.sh -r 1 -m sshleifer/tiny-distilbert-base-uncased-finetuned-sst-2-english -t MNLI -s run_glue.py -p ICX -L log/sshleifer/tiny-distilbert-base-uncased-finetuned-sst-2-english -b 1 -B distilbert
#./run_throughput_mode_tl.sh -r 1 -m sshleifer/tiny-distilbert-base-uncased-finetuned-sst-2-english -t MNLI -s run_glue.py -p ICX -L log/sshleifer/tiny-distilbert-base-uncased-finetuned-sst-2-english -b 160 -B distilbert

#./run_throughput_mode_tl.sh -r 1 -m facebook/bart-large-mnli -t MNLI -s run_glue.py -p ICX -L log/facebook/bart-large-mnli -b 1 -B bart
#./run_throughput_mode_tl.sh -r 1 -m facebook/bart-large-mnli -t MNLI -s run_glue.py -p ICX -L log/facebook/bart-large-mnli -b 160 -B bart
