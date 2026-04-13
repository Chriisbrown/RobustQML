import os
from argparse import ArgumentParser

# Import from other modules
from data.ADdataset import DataSet
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    process_folders = { 'background' : {'path':'background_for_training','pretty':'SM background'},
                        'ato4l': {'path':'Ato4l_lepFilter_13TeV_filtered','pretty':'A -> 4l'},
                        'hChToTauNu': {'path':'hChToTauNu_13TeV_PU20_filtered','pretty':'H+ -> Tau Nu'},
                        'hToTauTau': {'path':'hToTauTau_13TeV_PU20_filtered','pretty':'h -> Tau Tau'},
                        'leptoquark': {'path':'leptoquark_LOWMASS_lepFilter_13TeV_filtered','pretty':'leptoquark'},

                        }
    #outdir = 'training_data/'  
    outdir = '/eos/user/c/cebrown/RobustQML/AD_dataset/processed/'  

    os.makedirs(outdir, exist_ok=True)
    print("Output directory:", outdir)
        
    input_directory = '/eos/user/c/cebrown/RobustQML/AD_dataset/'
    
    for iprocess, process in tqdm(enumerate(process_folders.keys())):
        os.makedirs(outdir + process, exist_ok=True)
        
        train = DataSet.fromOrginalH5(input_directory+process_folders[process]['path'])
        test = DataSet.fromOrginalH5(input_directory+process_folders[process]['path'])
        augment_test = test

        train_split, test_split = train_test_split(train.data_frame, test_size=0.4)

        train.data_frame = train_split
        test.data_frame = test_split
        
        train.save_h5(outdir+process+'/train')
        train.pretty_name = process_folders[process]['pretty']
        train.plot_inputs(outdir+process+'/train')

        augment_test_split, test_split = train_test_split(test.data_frame, test_size=0.5)

        test.data_frame = test_split
        augment_test.data_frame = augment_test_split

        test.save_h5(outdir+process+'/test')
        
        #augment_test.drop_a_soft_one('jet')
        augment_test.eta_smear()
        augment_test.pt_smear()
        augment_test.phi_smear()
        augment_test.pretty_name = process_folders[process]['pretty'] + ' augmented'
        augment_test.plot_inputs(outdir+process+'/augment')
        augment_test.save_h5(outdir+process+'/augment')
        
    blackbox = DataSet.fromOrginalH5(input_directory+'BlackBox_background_mix',with_labels=True)
    blackbox_augment = blackbox
    
    train_split, test_split = train_test_split(blackbox.data_frame, test_size=0.2)
    
    blackbox.data_frame = train_split
    blackbox_augment.data_frame = test_split
    
    blackbox.save_h5(outdir+'/blackbox/test')
    blackbox.pretty_name = "blackbox"
    blackbox.plot_inputs(outdir+'blackbox/test')
    
    blackbox_augment.eta_smear()
    blackbox_augment.pt_smear()
    blackbox_augment.phi_smear()
    blackbox_augment.pretty_name = "blackbox augmented"
    blackbox_augment.plot_inputs(outdir+'blackbox/augment')
    blackbox_augment.save_h5(outdir+'blackbox/augment')
        
        
        
        
        
     
