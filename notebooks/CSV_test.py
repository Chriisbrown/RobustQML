from data.CSVdataset import DataSet

full_CSV = DataSet.fromCSV('/shared/RobustQML/SUSY.csv')

print(full_CSV.data_frame.head())