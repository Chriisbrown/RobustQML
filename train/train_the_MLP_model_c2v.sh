#python train/train.py -y model/configs/MLPContrastiveEmbeddingModel.yaml -o minbias/ContrastiveEmbedding -s minbias 
#python plot/test.py -o minbias/ContrastiveEmbedding 
#python plot/plot_latent.py -o minbias/ContrastiveEmbedding 
#python augmentations/augment_test.py -o minbias/ContrastiveEmbedding 

# python train/train.py -y model/configs/MLPContrastiveEmbeddingModel.yaml -o QCD/ContrastiveEmbedding -s QCD 
# python plot/test.py -o QCD/TransformerEmbedding 
# python plot/plot_latent.py -o QCD/TransformerEmbedding 
# python augmentations/augment_test.py -o QCD/TransformerEmbedding 



# python train/train.py -y model/configs/EmbeddingPennyLaneQAEModel.yaml -o minbias/minbiasTrfEmbeddingQAE/ -s minbias -m minbias/TransformerEmbedding 
# python plot/test.py  -o minbias/minbiasTrfEmbeddingQAE/ -e 10000 -m minbias/TransformerEmbedding 
# python augmentations/augment_test.py -o minbias/minbiasTrfEmbeddingQAE/ -e 10000 -m minbias/TransformerEmbedding 
 
# python train/train.py -y model/configs/EmbeddingPennyLaneQAEModel.yaml -o minbias/QCDTrfEmbeddingQAE/ -s minbias -m QCD/TransformerEmbedding 
# python plot/test.py  -o minbias/QCDTrfEmbeddingQAE/ -e 10000 -m QCD/TransformerEmbedding 
# python augmentations/augment_test.py -o minbias/QCDTrfEmbeddingQAE/ -e 10000 -m QCD/TransformerEmbedding 



# python train/train.py -y model/configs/EmbeddingPennyLaneQAEModel.yaml -o QCD/minbiasTrfEmbeddingQAE/ -s QCD -m minbias/TransformerEmbedding 
# python plot/test.py  -o QCD/minbiasTrfEmbeddingQAE/ -e 10000 -m minbias/TransformerEmbedding 
# python augmentations/augment_test.py -o QCD/minbiasTrfEmbeddingQAE/ -e 10000 -m minbias/TransformerEmbedding 

# python train/train.py -y model/configs/EmbeddingPennyLaneQAEModel.yaml -o QCD/QCDTrfEmbeddingQAE/ -s QCD -m QCD/TransformerEmbedding 
# python plot/test.py  -o QCD/QCDTrfEmbeddingQAE/ -e 10000 -m QCD/TransformerEmbedding 
# python augmentations/augment_test.py -o QCD/QCDTrfEmbeddingQAE/ -e 10000 -m QCD/TransformerEmbedding 




python train/train.py -y model/configs/EmbeddingClassicalAEModel.yaml -o minbias/minbiasTrfEmbeddingCAE/ -s minbias -m C2V_Contrastive_Embedding/MLP/minbias/ContrastiveEmbedding 
python plot/test.py  -o minbias/minbiasTrfEmbeddingCAE/ -m C2V_Contrastive_Embedding/MLP/minbias/ContrastiveEmbedding 
python augmentations/augment_test.py -o minbias/minbiasTrfEmbeddingCAE/ -m C2V_Contrastive_Embedding/MLP/minbias/ContrastiveEmbedding 
 
python train/train.py -y model/configs/EmbeddingClassicalAEModel.yaml -o minbias/QCDTrfEmbeddingCAE/ -s minbias -m C2V_Contrastive_Embedding/MLP/QCD/ContrastiveEmbedding 
python plot/test.py  -o minbias/QCDTrfEmbeddingCAE/ -m C2V_Contrastive_Embedding/MLP/QCD/ContrastiveEmbedding 
python augmentations/augment_test.py -o minbias/QCDTrfEmbeddingCAE/  -m C2V_Contrastive_Embedding/MLP/QCD/ContrastiveEmbedding 


python train/train.py -y model/configs/EmbeddingClassicalAEModel.yaml -o QCD/minbiasTrfEmbeddingCAE/ -s QCD -m C2V_Contrastive_Embedding/MLP/minbias/ContrastiveEmbedding 
python plot/test.py  -o QCD/minbiasTrfEmbeddingCAE/ -m C2V_Contrastive_Embedding/MLP/minbias/ContrastiveEmbedding 
python augmentations/augment_test.py -o QCD/minbiasTrfEmbeddingCAE/ -m C2V_Contrastive_Embedding/MLP/minbias/ContrastiveEmbedding 

python train/train.py -y model/configs/EmbeddingClassicalAEModel.yaml -o QCD/QCDTrfEmbeddingCAE/ -s QCD -m C2V_Contrastive_Embedding/MLP/QCD/ContrastiveEmbedding 
python plot/test.py  -o QCD/QCDTrfEmbeddingCAE/  -m C2V_Contrastive_Embedding/MLP/QCD/ContrastiveEmbedding 
python augmentations/augment_test.py -o QCD/QCDTrfEmbeddingCAE/  -m C2V_Contrastive_Embedding/MLP/QCD/ContrastiveEmbedding 



