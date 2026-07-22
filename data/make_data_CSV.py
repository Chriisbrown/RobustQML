import os
from argparse import ArgumentParser

# Import from other modules
from data.CSVdataset import DataSet
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    #outdir = 'training_data/'  
    input_directory = '/scratch/RobustQML_Datasets/SUSY.csv'  
    outdir = '/shared/RobustQML_Datasets/SUSY_processed/'  

    os.makedirs(outdir, exist_ok=True)
    print("Output directory:", outdir)
    
        
    test_fraction = 0.6
    
    data_set = DataSet.fromCSV(filepath=input_directory)
    
    processes = {"background":0,"SUSY":1}
    


    
    for iprocess, process in tqdm(enumerate(processes)):
        process_data_frame = data_set.data_frame[data_set.data_frame['event_label']==iprocess]
        
        process_data_set = DataSet(process)
        process_test_data_set = DataSet(process+'_test')
        process_augment_data_set = DataSet(process+'_augment')
        
        train=process_data_frame.sample(frac=(1-test_fraction))
        test=process_data_frame.drop(train.index) 
        
        train.reset_index(inplace=True,drop=True)
        test.reset_index(inplace=True,drop=True)
    
        test_again =test.sample(frac=(test_fraction/2))
        augment=test.drop(test_again.index) 
        
        test_again.reset_index(inplace=True,drop=True)
        augment.reset_index(inplace=True,drop=True)
        
        process_test_data_set.data_frame = test_again
        process_augment_data_set.data_frame = augment
        process_data_set.data_frame = train  
              
        process_data_set.data_frame['index'] = range(0,len(process_data_set.data_frame))
        process_test_data_set.data_frame['index'] = range(0,len(process_test_data_set.data_frame))
        process_augment_data_set.data_frame['index'] = range(0,len(process_augment_data_set.data_frame))
        
        print(process_data_frame.describe())
        print(process_data_set.data_frame.describe())
        print(process_test_data_set.data_frame.describe())
        print(process_augment_data_set.data_frame.describe())
        
        process_data_set.plot_inputs(outdir+process+'/train')
        process_data_set.save_h5(outdir+process+'/train')

        process_test_data_set.plot_inputs(outdir+process+'/test')
        process_test_data_set.save_h5(outdir+process+'/test')
        
        #process_augment_data_set.drop_a_soft_one('jet')
        process_augment_data_set.eta_smear()
        process_augment_data_set.pt_smear()
        process_augment_data_set.phi_smear()
        process_augment_data_set.plot_inputs(outdir+process+'/augment')
        process_augment_data_set.save_h5(outdir+process+'/augment')
     
