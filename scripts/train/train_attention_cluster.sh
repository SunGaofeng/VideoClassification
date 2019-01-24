#export PYTHONPATH=../lib:$PYTHONPATH
python train.py --model-name="AttentionCluster" --config=../configs/attention_cluster.txt --epoch-num=6 \
                --valid-interval=100 --save-interval=100 --save-dir=../save --log-interval=5
