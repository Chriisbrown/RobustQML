# python train/train.py -y model/configs/VICRegModel.yaml -o minbias/VICReg -s minbias
# python plot/test.py -o minbias/VICReg
# python plot/plot_latent.py -o minbias/VICReg
# python augmentations/augment_test.py -o minbias/VICReg

# python train/train.py -y model/configs/VICRegModel.yaml -o QCD/VICReg -s QCD
# python plot/test.py -o QCD/VICReg
# python plot/plot_latent.py -o QCD/VICReg
# python augmentations/augment_test.py -o QCD/VICReg

python train/train.py -y model/configs/VICRegModel.yaml -o qcd_but/VICReg -s qcd_but
python plot/test.py -o qcd_but/VICReg
python plot/plot_latent.py -o qcd_but/VICReg
python augmentations/augment_test.py -o qcd_but/VICReg

# python train/train.py -y model/configs/EmbeddingPennyLaneQAEModel.yaml -o minbias/minbiasVCREmbeddingQAE/ -s minbias -m minbias/VICReg
# python plot/test.py  -o minbias/minbiasVCREmbeddingQAE/ -e 10000 -m minbias/VICReg
# python augmentations/augment_test.py -o minbias/minbiasVCREmbeddingQAE/ -e 10000 -m minbias/VICReg

# python train/train.py -y model/configs/EmbeddingPennyLaneQAEModel.yaml -o minbias/QCDVCREmbeddingQAE/ -s minbias -m QCD/VICReg
# python plot/test.py  -o minbias/QCDVCREmbeddingQAE/ -e 10000 -m QCD/VICReg
# python augmentations/augment_test.py -o minbias/QCDVCREmbeddingQAE/ -e 10000 -m QCD/VICReg


# python train/train.py -y model/configs/EmbeddingPennyLaneQAEModel.yaml -o QCD/minbiasVCREmbeddingQAE/ -s QCD -m minbias/VICReg
# python plot/test.py  -o QCD/minbiasVCREmbeddingQAE/ -e 10000 -m minbias/VICReg
# python augmentations/augment_test.py -o QCD/minbiasVCREmbeddingQAE/ -e 10000 -m minbias/VICReg

# python train/train.py -y model/configs/EmbeddingPennyLaneQAEModel.yaml -o QCD/QCDVCREmbeddingQAE/ -s QCD -m QCD/VICReg
# python plot/test.py  -o QCD/QCDVCREmbeddingQAE/ -e 10000 -m QCD/VICReg
# python augmentations/augment_test.py -o QCD/minbiasVCREmbeddingQAE/ -e 10000 -m QCD/VICReg

python train/train.py -y model/configs/EmbeddingPennyLaneQAEModel.yaml -o minbias/qcd_butVCREmbeddingQAE/ -s minbias -m qcd_but/VICReg
python plot/test.py  -o minbias/qcd_butVCREmbeddingQAE/ -e 10000 -m qcd_but/VICReg
python augmentations/augment_test.py -o minbias/qcd_butVCREmbeddingQAE/ -e 10000 -m qcd_but/VICReg

python train/train.py -y model/configs/EmbeddingPennyLaneQAEModel.yaml -o QCD/acd_butVCREmbeddingQAE/ -s QCD -m qcd_but/VICReg
python plot/test.py  -o QCD/qcd_butVCREmbeddingQAE/ -e 10000 -m qcd_but/VICReg
python augmentations/augment_test.py -o QCD/qcd_butVCREmbeddingQAE/ -e 10000 -m qcd_but/VICReg


python train/train.py -y model/configs/EmbeddingPennyLaneQAEModel.yaml -o qcd_but/qcd_butVCREmbeddingQAE/ -s QCD -m qcd_but/VICReg
python plot/test.py  -o qcd_but/qcd_butVCREmbeddingQAE/ -e 10000 -m qcd_but/VICReg
python augmentations/augment_test.py -o qcd_but/qcd_butVCREmbeddingQAE/ -e 10000 -m qcd_but/VICReg





