export PYTHONPATH=../:$PYTHONPATH
python test.py --model-name="AttentionCluster" --config=./configs/attention_cluster.txt \
                --log-interval=5
