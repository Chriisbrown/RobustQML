python run_full_model.py -a -e -p -aug -pl -s -m MLP -d AD &> output_MLP_AD.txt
python run_full_model.py -p -aug -pl -s -m MLP -d C2V > output_MLP_C2V.txt
python run_full_model.py -e -a -p -aug -pl -s -m transformer -d AD &> output_Transformer_AD.txt
python run_full_model.py -p -aug -pl -s -m transformer -d C2V &> output_Transformer_C2V.txt
