import os
from argparse import ArgumentParser

# Import from other modules
from data.dataset import DataSet

if __name__ == "__main__":

    parser = ArgumentParser()
    # Making input arguments
    parser.add_argument(
        '-i',
        '--input',
        default='WJetsToLNu',
        help='Name of input file',
    )
    
    parser.add_argument(
        '-n',
        '--num_events',
        default=200,
        help='Number of events to process',
        type=int,
    )

    args = parser.parse_args()
    
    data_set = DataSet.fromHF(args.input,max_number_of_events=args.num_events)
    data_set.plot_inputs(args.input)
    data_set.save_h5(args.input)
    