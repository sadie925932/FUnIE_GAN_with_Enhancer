python train_funiegan.py \
--cfg_file configs/train_uieb.yaml \
--epoch 0 --num_epochs 201 \
--attention_mode Spatial \
--pyramid_mode Try

python train_funiegan.py \
--cfg_file configs/train_uieb.yaml \
--epoch 0 --num_epochs 201 \
--attention_mode Both \
--pyramid_mode Try

