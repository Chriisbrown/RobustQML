python train/train.py -y model/configs/AutoEncoderModel.yaml -o output/autoencoder
python plot/test.py -o output/autoencoder
python augmentations/augment_test.py -o output/autoencoder

python train/train.py -y model/configs/AXOVariationalAutoEncoderModel.yaml -o output/AXOVariationalAutoEncoder
python plot/test.py -o output/AXOVariationalAutoEncoder
python augmentations/augment_test.py -o output/AXOVariationalAutoEncoder

python train/train.py -y model/configs/VariationalAutoEncoderModel.yaml -o output/VariationalAutoEncoder
python plot/test.py -o output/VariationalAutoEncoder
python augmentations/augment_test.py -o output/VariationalAutoEncoder

python train/train.py -y model/configs/IsolationTreeModel.yaml -o output/IsolationTree
python plot/test.py -o output/IsolationTree
python augmentations/augment_test.py -o output/IsolationTree

python train/train.py -y model/configs/VICRegModel.yaml -o output/VICReg
python plot/test.py -o output/VICReg
python augmentations/augment_test.py -o output/VICReg

python train/train.py -y model/configs/ContrastiveEmbeddingModel.yaml -o output/ContrastiveEmbedding
python plot/test.py -o output/ContrastiveEmbedding
python augmentations/augment_test.py -o output/ContrastiveEmbedding

python train/train.py -y model/configs/ContrastiveEmbeddingModel_sup.yaml -o output/ContrastiveEmbedding_sup
python plot/test.py -o output/ContrastiveEmbedding_sup
python augmentations/augment_test.py -o output/ContrastiveEmbedding_sup

