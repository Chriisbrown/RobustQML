import os
from argparse import ArgumentParser

# Import from other modules
from data.EOSdataset import DataSet
from tqdm import tqdm
import pandas as pd

if __name__ == "__main__":

    process_folders = { 'HH_4b' :'HH_4b-NEVENT10000-RS20',
                    #    'HH_bbgammagamma': 'HH_bbgammagamma-NEVENT10000-RS21',
                    #    'HH_bbtautau': 'HH_bbtautau-NEVENT10000-RS22',
                         'QCD_HT50toInf': 'QCD_HT50toInf-NEVENT10000-RS26',
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
                         'minbias': 'minbias-NEVENT10000-RS59',
                         'upsilon_to_leptons' : 'upsilon_to_leptons-NEVENT10000-RS60'
                        }
    outdir = 'training_data/'  

    os.makedirs(outdir, exist_ok=True)
    print("Output directory:", outdir)
    
    
    
    max_files_per_process = 100
    
    test_fraction = 0.1
    
    
    input_directory = '/eos/project/f/foundational-model-dataset/samples/production_final/'
    
    for iprocess, process in tqdm(enumerate(process_folders.keys())):
        data_set_list = []
        os.makedirs(outdir + process, exist_ok=True)
        for ifile in tqdm(range(max_files_per_process)):
            input_file = input_directory + process + '/' + process_folders[process] + str(ifile+1).zfill(6) + '.parquet'
            
            data_set = DataSet.fromEOS(filepath=input_file)
            data_set_list.append(data_set)
    
        process_data_frame = pd.concat([data.data_frame for data in data_set_list])
        process_data_set = DataSet(process)
        process_data_set.data_frame = process_data_frame
        process_data_set.plot_inputs(outdir+process)
        process_data_set.save_h5(outdir+process)
    