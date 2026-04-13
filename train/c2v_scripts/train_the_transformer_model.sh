#python train/train.py -y model/configs/TransformerContrastiveEmbeddingModel.yaml -o minbias/TransformerContrastiveEmbedding -s minbias
# python plot/test.py -o minbias/TransformerContrastiveEmbedding
# python plot/plot_latent.py -o minbias/TransformerContrastiveEmbedding
# python augmentations/augment_test.py -o minbias/TransformerContrastiveEmbedding

# python train/train.py -y model/configs/TransformerContrastiveEmbeddingModel.yaml -o QCD/TransformerContrastiveEmbedding -s QCD
# python plot/test.py -o QCD/TransformerContrastiveEmbedding
# python plot/plot_latent.py -o QCD/TransformerContrastiveEmbedding
# python augmentations/augment_test.py -o QCD/TransformerContrastiveEmbedding

python train/train.py -y model/configs/TransformerContrastiveEmbeddingModel.yaml -o qcd_but/TransformerContrastiveEmbedding -s qcd_but
python plot/test.py -o qcd_but/TransformerContrastiveEmbedding
python plot/plot_latent.py -o qcd_but/TransformerContrastiveEmbedding
python augmentations/augment_test.py -o qcd_but/TransformerContrastiveEmbedding


#python train/train.py -y model/configs/EmbeddingPennyLaneQAEModel.yaml -o minbias/minbiasTrfEmbeddingQAE/ -s minbias -m minbias/TransformerContrastiveEmbedding
# python plot/test.py  -o minbias/minbiasTrfEmbeddingQAE/ -e 10000 -m minbias/TransformerContrastiveEmbedding
# python augmentations/augment_test.py -o minbias/minbiasTrfEmbeddingQAE/ -e 10000 -m minbias/TransformerContrastiveEmbedding

# #python train/train.py -y model/configs/EmbeddingPennyLaneQAEModel.yaml -o minbias/QCDTrfEmbeddingQAE/ -s minbias -m QCD/TransformerContrastiveEmbedding
# python plot/test.py  -o minbias/QCDTrfEmbeddingQAE/ -e 10000 -m QCD/TransformerContrastiveEmbedding
# python augmentations/augment_test.py -o minbias/minbiasTrfEmbeddingQAE/ -e 10000 -m QCD/TransformerContrastiveEmbedding

python train/train.py -y model/configs/EmbeddingPennyLaneQAEModel.yaml -o minbias/qcd_butTrfEmbeddingQAE/ -s minbias -m qcd_but/TransformerContrastiveEmbedding
python plot/test.py  -o minbias/qcd_butTrfEmbeddingQAE/ -e 10000 -m qcd_but/TransformerContrastiveEmbedding
python augmentations/augment_test.py -o minbias/qcd_butTrfEmbeddingQAE/ -e 10000 -m qcd_but/TransformerContrastiveEmbedding


#python train/train.py -y model/configs/EmbeddingPennyLaneQAEModel.yaml -o QCD/minbiasTrfEmbeddingQAE/ -s QCD -m minbias/TransformerContrastiveEmbedding
# python plot/test.py  -o QCD/minbiasTrfEmbeddingQAE/ -e 10000 -m minbias/TransformerContrastiveEmbedding
# python augmentations/augment_test.py -o QCD/minbiasTrfEmbeddingQAE/ -e 10000 -m minbias/TransformerContrastiveEmbedding

# #python train/train.py -y model/configs/EmbeddingPennyLaneQAEModel.yaml -o QCD/QCDTrfEmbeddingQAE/ -s QCD -m QCD/TransformerContrastiveEmbedding
# python plot/test.py  -o QCD/QCDTrfEmbeddingQAE/ -e 10000 -m QCD/TransformerContrastiveEmbedding
# python augmentations/augment_test.py -o QCD/minbiasTrfEmbeddingQAE/ -e 10000 -m QCD/TransformerContrastiveEmbedding

python train/train.py -y model/configs/EmbeddingPennyLaneQAEModel.yaml -o QCD/qcd_butTrfEmbeddingQAE/ -s QCD -m qcd_but/TransformerContrastiveEmbedding
python plot/test.py  -o QCD/qcd_butTrfEmbeddingQAE/ -e 10000 -m qcd_but/TransformerContrastiveEmbedding
python augmentations/augment_test.py -o QCD/qcd_butTrfEmbeddingQAE/ -e 10000 -m qcd_but/TransformerContrastiveEmbedding


# python train/train.py -y model/configs/QCDEmbeddingPennyLaneQAEModel.yaml -o all/minbiasTrfEmbeddingQAE/ -s all -m minbias/TransformerContrastiveEmbedding
# python plot/test.py  -o QCD/minbiasTrfEmbeddingQAE/ -e 10000
# python augmentations/augment_test.py -o QCD/minbiasTrfEmbeddingQAE/ -e 10000

# python train/train.py -y model/configs/QCDEmbeddingPennyLaneQAEModel.yaml -o all/QCDTrfEmbeddingQAE/ -s all -m QCD/TransformerContrastiveEmbedding
# python plot/test.py  -o all/QCDTrfEmbeddingQAE/ -e 10000
# python augmentations/augment_test.py -o all/minbiasTrfEmbeddingQAE/ -e 10000

python train/train.py -y model/configs/EmbeddingPennyLaneQAEModel.yaml -o qcd_but/qcd_butTrfEmbeddingQAE/ -s qcd_but -m qcd_but/TransformerContrastiveEmbedding
python plot/test.py  -o qcd_but/qcd_butTrfEmbeddingQAE/ -e 10000 -m qcd_but/TransformerContrastiveEmbedding
python augmentations/augment_test.py -o qcd_but/qcd_butTrfEmbeddingQAE/ -e 10000 -m qcd_but/TransformerContrastiveEmbedding
 



