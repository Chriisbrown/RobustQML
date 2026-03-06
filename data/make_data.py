import os
from argparse import ArgumentParser

# Import from other modules
from data.EOSdataset import DataSet
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    process_folders = { #'HH_4b' :{'prefix':'HH_4b-NEVENT10000-RS20','n_events':200},
                        #'HH_bbgammagamma': {'prefix':'HH_bbgammagamma-NEVENT10000-RS21','n_events':200},
                        #'HH_bbtautau': {'prefix':'HH_bbtautau-NEVENT10000-RS22','n_events':200},
                        #'QCD_HT50toInf': {'prefix':'QCD_HT50toInf-NEVENT10000-RS26','n_events':5000},
                        #'QCD_HT50tobb':{'prefix':'QCD_HT50tobb-NEVENT10000-RS25','n_events':200},
                    #    'tt0123j_5f_ckm_LO_MLM_hadronic':'tt0123j_5f_ckm_LO_MLM_hadronic-NEVENT10000-RS28',
                    #    'ggHgluglu':'ggHgluglu-NEVENT10000-RS16',
                    #    'ggHtautau':'ggHtautau-NEVENT10000-RS17',
                    #    'ggHbb':'ggHbb-NEVENT10000-RS13',
                    #    'ggHcc':'ggHcc-NEVENT10000-RS14',
                    #    'VBFHbb':'VBFHbb-NEVENT10000-RS35',
                    #    'VBFHcc':'VBFHcc-NEVENT10000-RS36',
                    #    'VBFHgammagamma':'VBFHgammagamma-NEVENT10000-RS37',
                    #    'VBFHgluglu':'VBFHgluglu-NEVENT10000-RS38',
                    #    'VBFHtautau':'VBFHtautau-NEVENT10000-RS39',
                    #    'WJetsToLNu_13TeV-madgraphMLM-pythia8':'WJetsToLNu_13TeV-madgraphMLM-pythia8-NEVENT10000-RS44'
                         'minbias': {'prefix':'minbias-NEVENT10000-RS59','n_events':5000},
                    #     'upsilon_to_leptons' : {'prefix':'upsilon_to_leptons-NEVENT10000-RS60','n_events':200},
                        }
    #outdir = 'training_data/'  
    outdir = '/eos/user/c/cebrown/RobustQML/training_data/'  

    os.makedirs(outdir, exist_ok=True)
    print("Output directory:", outdir)
    
        
    test_fraction = 0.6
    
    input_directory = '/eos/project/f/foundational-model-dataset/samples/production_final/'
    
    for iprocess, process in tqdm(enumerate(process_folders.keys())):
        data_set_list = []
        os.makedirs(outdir + process, exist_ok=True)
        for ifile in tqdm(range(process_folders[process]['n_events'])):
            input_file = input_directory + process + '/' + process_folders[process]['prefix'] + str(ifile+1).zfill(6) + '.parquet'
            
            data_set = DataSet.fromEOS(filepath=input_file)
            data_set_list.append(data_set)
    
        process_data_frame = pd.concat([data.data_frame for data in data_set_list])
        process_data_frame.reset_index(inplace=True,drop=True)
        process_data_frame.index = range(0,len(process_data_frame))
        
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
              
        process_data_set.generate_feature_lists()
        process_data_set.data_frame['index'] = range(0,len(process_data_set.data_frame))
        process_test_data_set.generate_feature_lists()
        process_test_data_set.data_frame['index'] = range(0,len(process_test_data_set.data_frame))
        process_augment_data_set.generate_feature_lists()
        process_augment_data_set.data_frame['index'] = range(0,len(process_augment_data_set.data_frame))
        
        print(process_data_frame.describe())
        print(process_data_set.data_frame.describe())
        print(process_test_data_set.data_frame.describe())
        print(process_augment_data_set.data_frame.describe())
        
        process_data_set.plot_inputs(outdir+process+'/train')
        process_data_set.save_h5(outdir+process+'/train')

        process_test_data_set.plot_inputs(outdir+process+'/test')
        process_test_data_set.save_h5(outdir+process+'/test')
        
        process_augment_data_set.drop_a_soft_one('jet')
        process_augment_data_set.eta_smear()
        process_augment_data_set.pt_smear()
        process_augment_data_set.phi_smear()
        process_augment_data_set.plot_inputs(outdir+process+'/augment')
        process_augment_data_set.save_h5(outdir+process+'/augment')
     
