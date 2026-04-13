#python train/train.py -y model/configs/ContrastiveEmbeddingModel_sup.yaml -o background/ContrastiveEmbedding -s background --ad_dataset
#python plot/test.py -o background/ContrastiveEmbedding --ad_dataset
#python plot/plot_latent.py -o background/ContrastiveEmbedding --ad_dataset
#python augmentations/augment_test.py -o background/ContrastiveEmbedding --ad_dataset

#python train/train.py -y model/configs/ContrastiveEmbeddingModel_sup.yaml -o all/ContrastiveEmbedding -s all --ad_dataset
#python plot/test.py -o all/ContrastiveEmbedding --ad_dataset
#python plot/plot_latent.py -o all/ContrastiveEmbedding --ad_dataset
#python augmentations/augment_test.py -o all/ContrastiveEmbedding --ad_dataset



#python train/train.py -y model/configs/EmbeddingPennyLaneQAEModel.yaml -o background/backgroundConEmbeddingQAE/ -s background -m background/ContrastiveEmbedding --ad_dataset
#python plot/test.py  -o background/backgroundConEmbeddingQAE/ -e 10000 -m background/ContrastiveEmbedding --ad_dataset
#python augmentations/augment_test.py -o background/backgroundConEmbeddingQAE/ -e 10000 -m background/ContrastiveEmbedding --ad_dataset
 
#python train/train.py -y model/configs/EmbeddingPennyLaneQAEModel.yaml -o background/allConEmbeddingQAE/ -s background -m all/ContrastiveEmbedding --ad_dataset
#python plot/test.py  -o background/allConEmbeddingQAE/ -e 10000 -m all/ContrastiveEmbedding --ad_dataset
#python augmentations/augment_test.py -o background/allConEmbeddingQAE/ -e 10000 -m all/ContrastiveEmbedding --ad_dataset



#python train/train.py -y model/configs/EmbeddingPennyLaneQAEModel.yaml -o all/backgroundConEmbeddingQAE/ -s all -m background/ContrastiveEmbedding --ad_dataset
#python plot/test.py  -o all/backgroundConEmbeddingQAE/ -e 10000 -m background/ContrastiveEmbedding --ad_dataset
#python augmentations/augment_test.py -o all/backgroundConEmbeddingQAE/ -e 10000 -m background/ContrastiveEmbedding --ad_dataset

#python train/train.py -y model/configs/EmbeddingPennyLaneQAEModel.yaml -o all/allConEmbeddingQAE/ -s all -m all/ContrastiveEmbedding --ad_dataset
python plot/test.py  -o all/allConEmbeddingQAE/ -e 10000 -m all/ContrastiveEmbedding --ad_dataset
python augmentations/augment_test.py -o all/allConEmbeddingQAE/ -e 10000 -m all/ContrastiveEmbedding --ad_dataset




 



