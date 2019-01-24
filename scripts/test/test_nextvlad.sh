export PYTHONPATH=../:$PYTHONPATH
python test.py --model-name="NEXTVLAD" --config=./configs/nextvlad.txt \
                --log-interval=5 --weights=./save/NEXTVLAD_epoch0
