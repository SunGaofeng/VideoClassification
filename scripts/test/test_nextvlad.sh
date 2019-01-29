export PYTHONPATH=../:$PYTHONPATH
python test.py --model-name="NEXTVLAD" --config=./configs/nextvlad.txt \
                --log-interval=10 --weights=/home/sungaofeng/programs/nextvlad/dev_paddle/save/model_epoch4_end
