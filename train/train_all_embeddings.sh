# python train/train.py -y model/configs/ContrastiveEmbeddingModel_sup.yaml -o embeddings/AD/background/MLPEmbedding -s background --ad_dataset
# python plot/plot_latent.py -o embeddings/AD/background/MLPEmbedding --ad_dataset

# python train/train.py -y model/configs/ContrastiveEmbeddingModel_sup.yaml -o embeddings/AD/all/MLPEmbedding -s all --ad_dataset
# python plot/plot_latent.py -o embeddings/AD/all/MLPEmbedding --ad_dataset

python train/train.py -y model/configs/ContrastiveEmbeddingModel_sup.yaml -o embeddings/C2V/minbias/MLPEmbedding -s minbias 
python plot/plot_latent.py -o  embeddings/C2V/minbias/MLPEmbedding 

python train/train.py -y model/configs/ContrastiveEmbeddingModel_sup.yaml -o embeddings/C2V/QCD/MLPEmbedding -s QCD 
python plot/plot_latent.py -o embeddings/C2V/QCD/MLPEmbedding 

python train/train.py -y model/configs/ContrastiveEmbeddingModel_sup.yaml -o embeddings/C2V/all/MLPEmbedding -s all 
python plot/plot_latent.py -o embeddings/C2V/all/MLPEmbedding 

# python train/train.py -y model/configs/TransformerContrastiveEmbeddingModel.yaml -o embeddings/AD/background/TransformerEmbedding -s background --ad_dataset
# python plot/plot_latent.py -o embeddings/AD/background/TransformerEmbedding --ad_dataset

# python train/train.py -y model/configs/TransformerContrastiveEmbeddingModel.yaml -o embeddings/AD/all/TransformerEmbedding -s all --ad_dataset
# python plot/plot_latent.py -o embeddings/AD/all/TransformerEmbedding --ad_dataset

# python train/train.py -y model/configs/TransformerContrastiveEmbeddingModel.yaml -o embeddings/C2V/minbias/TransformerEmbedding -s minbias 
# python plot/plot_latent.py -o embeddings/C2V/minbias/TransformerEmbedding 

# python train/train.py -y model/configs/TransformerContrastiveEmbeddingModel.yaml -o embeddings/C2V/QCD/TransformerEmbedding -s QCD 
python plot/plot_latent.py -o embeddings/C2V/QCD/TransformerEmbedding 

python train/train.py -y model/configs/TransformerContrastiveEmbeddingModel.yaml -o embeddings/C2V/all/TransformerEmbedding -s all 
python plot/plot_latent.py -o embeddings/C2V/all/TransformerEmbedding 
