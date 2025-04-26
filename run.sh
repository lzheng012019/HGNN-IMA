
#amazon
python train_modal2_little.py --use_norm True --device 1 --num_layers 3 --epoch 100 --num_heads 8 --feats-type 0 --dataset AMAZON --type modal2_missing
#imdb
python train_modal2_little.py --use_norm True --num_layers 3 --epoch 100 --num_heads 8 --feats-type 0 --dataset IMDB --type modal2_missing
 #douban
python train_modal2_little.py --use_norm True --num_layers 3 --epoch 50 --num_heads 8 --feats-type 0 --dataset DOUBAN --type modal2_missing
#amazon1
python train_modal2.py --use_norm True --device 1 --num_layers 3 --epoch 200 --num_heads 8 --feats-type 0 --dataset AMAZON1 --type modal2_align
#amazon2
python train_modal2.py --use_norm True --device 1 --num_layers 3 --epoch 200 --num_heads 8 --feats-type 0 --dataset AMAZON1 --type modal2_align
python train_modal3.py --use_norm True --device 1 --num_layers 3 --epoch 200 --num_heads 8 --feats-type 0 --dataset AMAZON2 --type modal2_align
