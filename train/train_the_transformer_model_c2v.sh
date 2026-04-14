#python train/train.py -y model/configs/TransformerContrastiveEmbeddingModel.yaml -o minbias/TransformerEmbedding -s minbias 
#python plot/test.py -o minbias/TransformerEmbedding 
#python plot/plot_latent.py -o minbias/TransformerEmbedding 
#python augmentations/augment_test.py -o minbias/TransformerEmbedding 

# python train/train.py -y model/configs/TransformerContrastiveEmbeddingModel.yaml -o QCD/TransformerEmbedding -s QCD 
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




#python train/train.py -y model/configs/EmbeddingClassicalAEModel.yaml -o minbias/minbiasTrfEmbeddingCAE/ -s minbias -m C2V_Contrastive_Embedding/Transformer/minbias/TransformerEmbedding 
#python plot/test.py  -o minbias/minbiasTrfEmbeddingCAE/ -m C2V_Contrastive_Embedding/Transformer/minbias/TransformerEmbedding 
#python augmentations/augment_test.py -o minbias/minbiasTrfEmbeddingCAE/ -m C2V_Contrastive_Embedding/Transformer/minbias/TransformerEmbedding 
 
#python train/train.py -y model/configs/EmbeddingClassicalAEModel.yaml -o minbias/QCDTrfEmbeddingCAE/ -s minbias -m C2V_Contrastive_Embedding/Transformer/QCD/TransformerEmbedding 
# python plot/test.py  -o minbias/QCDTrfEmbeddingCAE/ -m C2V_Contrastive_Embedding/Transformer/QCD/TransformerEmbedding 
# python augmentations/augment_test.py -o minbias/QCDTrfEmbeddingCAE/  -m C2V_Contrastive_Embedding/Transformer/QCD/TransformerEmbedding 


#python train/train.py -y model/configs/EmbeddingClassicalAEModel.yaml -o QCD/minbiasTrfEmbeddingCAE/ -s QCD -m C2V_Contrastive_Embedding/Transformer/minbias/TransformerEmbedding 
#python plot/test.py  -o QCD/minbiasTrfEmbeddingCAE/ -m C2V_Contrastive_Embedding/Transformer/minbias/TransformerEmbedding 
python augmentations/augment_test.py -o QCD/minbiasTrfEmbeddingCAE/ -m C2V_Contrastive_Embedding/Transformer/minbias/TransformerEmbedding 

python train/train.py -y model/configs/EmbeddingClassicalAEModel.yaml -o QCD/QCDTrfEmbeddingCAE/ -s QCD -m C2V_Contrastive_Embedding/Transformer/QCD/TransformerEmbedding 
python plot/test.py  -o QCD/QCDTrfEmbeddingCAE/  -m C2V_Contrastive_Embedding/Transformer/QCD/TransformerEmbedding 
python augmentations/augment_test.py -o QCD/QCDTrfEmbeddingCAE/  -m C2V_Contrastive_Embedding/Transformer/QCD/TransformerEmbedding 



