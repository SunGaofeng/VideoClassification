export PYTHONPATH=../lib:$PYTHONPATH
python train.py --model-name="NEXTVLAD" --config=../configs/nextvlad.txt --epoch-num=6 \
                --valid-interval=100 --save-interval=100 --save-dir=../save --log-interval=5
