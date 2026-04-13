# python train/train.py -y model/configs/AutoEncoderModel.yaml -o output/autoencoder
# python plot/test.py -o output/autoencoder
# python augmentations/augment_test.py -o output/autoencoder

# python train/train.py -y model/configs/AXOVariationalAutoEncoderModel.yaml -o output/AXOVariationalAutoEncoder
# python plot/test.py -o output/AXOVariationalAutoEncoder
# python augmentations/augment_test.py -o output/AXOVariationalAutoEncoder

# python train/train.py -y model/configs/VariationalAutoEncoderModel.yaml -o output/VariationalAutoEncoder
# python plot/test.py -o output/VariationalAutoEncoder
# python augmentations/augment_test.py -o output/VariationalAutoEncoder

# python train/train.py -y model/configs/IsolationTreeModel.yaml -o output/IsolationTree
# python plot/test.py -o output/IsolationTree
# python augmentations/augment_test.py -o output/IsolationTree

# python train/train.py -y model/configs/VICRegModel.yaml -o minbias_output/VICReg -s minbias
# python plot/test.py -o minbias_output/VICReg
# python augmentations/augment_test.py -o minbias_output/VICReg

# python train/train.py -y model/configs/ContrastiveEmbeddingModel.yaml -o minbias_output/ContrastiveEmbedding -s minbias
# python plot/test.py -o minbias_output/ContrastiveEmbedding
# python augmentations/augment_test.py -o minbias_output/ContrastiveEmbedding 

# python train/train.py -y model/configs/ContrastiveEmbeddingModel_sup.yaml -o minbias_output/ContrastiveEmbedding_sup -s minbias
# python plot/test.py -o minbias_output/ContrastiveEmbedding_sup
# python augmentations/augment_test.py -o minbias_output/ContrastiveEmbedding_sup

# python train/train.py -y model/configs/TransformerContrastiveEmbeddingModel.yaml -o minbias_output/TransformerContrastiveEmbedding -s minbias
# python plot/test.py -o minbias_output/TransformerContrastiveEmbedding
# python augmentations/augment_test.py -o minbias_output/TransformerContrastiveEmbedding

# python train/train.py -y model/configs/VICRegModel.yaml -o QCD_output/VICReg -s QCD
# python plot/test.py -o QCD_output/VICReg
# python augmentations/augment_test.py -o QCD_output/VICReg

# python train/train.py -y model/configs/ContrastiveEmbeddingModel.yaml -o QCD_output/ContrastiveEmbedding -s QCD
# python plot/test.py -o QCD_output/ContrastiveEmbedding
# python augmentations/augment_test.py -o QCD_output/ContrastiveEmbedding 

# python train/train.py -y model/configs/ContrastiveEmbeddingModel_sup.yaml -o QCD_output/ContrastiveEmbedding_sup -s QCD
# python plot/test.py -o QCD_output/ContrastiveEmbedding_sup
# python augmentations/augment_test.py -o QCD_output/ContrastiveEmbedding_sup

# python train/train.py -y model/configs/TransformerContrastiveEmbeddingModel.yaml -o QCD_output/TransformerContrastiveEmbedding -s QCD
# python plot/test.py -o QCD_output/TransformerContrastiveEmbedding
# python augmentations/augment_test.py -o QCD_output/TransformerContrastiveEmbedding

python train/train.py -y model/configs/VICRegModel.yaml -o all_output/VICReg -s all
python plot/test.py -o all_output/VICReg
python plot/plot_latent.py -o all_output/VICReg
python augmentations/augment_test.py -o all_output/VICReg

# python train/train.py -y model/configs/ContrastiveEmbeddingModel.yaml -o all_output/ContrastiveEmbedding -s all
# python plot/test.py -o all_output/ContrastiveEmbedding
# python plot/plot_latent.py -o all_output/ContrastiveEmbedding
# python augmentations/augment_test.py -o all_output/ContrastiveEmbedding 

python train/train.py -y model/configs/ContrastiveEmbeddingModel_sup.yaml -o all_output/ContrastiveEmbedding_sup -s all
python plot/test.py -o all_output/ContrastiveEmbedding_sup
python plot/plot_latent.py -o all_output/ContrastiveEmbedding_sup
python augmentations/augment_test.py -o all_output/ContrastiveEmbedding_sup

python train/train.py -y model/configs/TransformerContrastiveEmbeddingModel.yaml -o all_output/TransformerContrastiveEmbedding -s all
python plot/test.py -o all_output/TransformerContrastiveEmbedding
python plot/plot_latent.py -o all_output/TransformerContrastiveEmbedding
python augmentations/augment_test.py -o all_output/TransformerContrastiveEmbedding

# python train/train.py -y model/configs/minbiasEmbeddingPennyLaneQAEModel.yaml -o minbias_output/minbiasTrfEmbeddingQAE/ -s minbias
# python plot/test.py  -o minbias_output/minbiasTrfEmbeddingQAE/ -e 10000
# python augmentations/augment_test.py -o minbias_output/minbiasTrfEmbeddingQAE/ -e 10000




