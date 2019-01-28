export PYTHONPATH=../:$PYTHONPATH
python test.py --model-name="AttentionLSTM" --config=./configs/attention_lstm.txt \
                --log-interval=5
