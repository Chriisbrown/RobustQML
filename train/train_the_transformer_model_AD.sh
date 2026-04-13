python train/train.py -y model/configs/TransformerContrastiveEmbeddingModel.yaml -o background/TransformerEmbedding -s background --ad_dataset
python plot/test.py -o background/TransformerEmbedding --ad_dataset
python plot/plot_latent.py -o background/TransformerEmbedding --ad_dataset
python augmentations/augment_test.py -o background/TransformerEmbedding --ad_dataset

python train/train.py -y model/configs/TransformerContrastiveEmbeddingModel.yaml -o all/TransformerEmbedding -s all --ad_dataset
python plot/test.py -o all/TransformerEmbedding --ad_dataset
python plot/plot_latent.py -o all/TransformerEmbedding --ad_dataset
python augmentations/augment_test.py -o all/TransformerEmbedding --ad_dataset



python train/train.py -y model/configs/EmbeddingPennyLaneQAEModel.yaml -o background/backgroundTrfEmbeddingQAE/ -s background -m background/TransformerEmbedding --ad_dataset
python plot/test.py  -o background/backgroundTrfEmbeddingQAE/ -e 10000 -m background/TransformerEmbedding --ad_dataset
python augmentations/augment_test.py -o background/backgroundTrfEmbeddingQAE/ -e 10000 -m background/TransformerEmbedding --ad_dataset
 
python train/train.py -y model/configs/EmbeddingPennyLaneQAEModel.yaml -o background/allTrfEmbeddingQAE/ -s background -m all/TransformerEmbedding --ad_dataset
python plot/test.py  -o background/allTrfEmbeddingQAE/ -e 10000 -m all/TransformerEmbedding --ad_dataset
python augmentations/augment_test.py -o background/allTrfEmbeddingQAE/ -e 10000 -m all/TransformerEmbedding --ad_dataset



python train/train.py -y model/configs/EmbeddingPennyLaneQAEModel.yaml -o all/backgroundTrfEmbeddingQAE/ -s all -m background/TransformerEmbedding --ad_dataset
python plot/test.py  -o all/backgroundTrfEmbeddingQAE/ -e 10000 -m background/TransformerEmbedding --ad_dataset
python augmentations/augment_test.py -o all/backgroundTrfEmbeddingQAE/ -e 10000 -m background/TransformerEmbedding --ad_dataset

python train/train.py -y model/configs/EmbeddingPennyLaneQAEModel.yaml -o all/allTrfEmbeddingQAE/ -s all -m all/TransformerEmbedding --ad_dataset
python plot/test.py  -o all/allTrfEmbeddingQAE/ -e 10000 -m all/TransformerEmbedding --ad_dataset
python augmentations/augment_test.py -o all/allTrfEmbeddingQAE/ -e 10000 -m all/TransformerEmbedding --ad_dataset




 



