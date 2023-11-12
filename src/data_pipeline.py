import pandas as pd
import utils


# Import functions
def read_data(return_file=False):
    """Read the history data"""
    # Read & parse data
    data = pd.read_csv(CONFIG_FILE['raw_dataset_path'],
                       parse_dates = [CONFIG_FILE['datetime_col']])

    # Extract metadata
    features_col = data.columns.tolist()
    classes_col = [col_class.name for col_class in data.dtypes.tolist()]
    contents_col = [data[col].head(3).tolist() for col in features_col]
    
    # Validate
    print('Data shape       :', data.shape)
    print('Features columns :', features_col)
    print('Class columns    :', classes_col)
    print('Content columns  :', contents_col)

    # Dump data
    utils.pickle_dump(data, CONFIG_FILE['dataset_path'])

    if return_file:
        return data
    
def split_train_test(return_file=False):
    """Split train & test data"""
    # Load data
    data = utils.pickle_load(CONFIG_FILE['dataset_path'])
    
    # Get the train data conditions
    cond_train = data[CONFIG_FILE['datetime_col']] <= CONFIG_FILE['last_date_train']

    # Split the data
    data_train = data[cond_train]
    data_test = data[~cond_train]

    # Validate the data
    print('Data train shape :', data_train.shape)
    print('Data test shape  :', data_test.shape)

    # Dump the data
    utils.pickle_dump(data_train, CONFIG_FILE['dataset_train_path'])
    utils.pickle_dump(data_test, CONFIG_FILE['dataset_test_path'])

    if return_file:
        return data_train, data_test

def split_input_ouput(type, return_file=False):
    """Split input (X) and output (y)"""
    # Load data
    data = utils.pickle_load(CONFIG_FILE[f'dataset_{type}_path'])
    
    # Split input & output
    y = data[CONFIG_FILE['target_col']]
    X = data.drop(columns=[CONFIG_FILE['target_col']], axis=1)
    
    # validate
    print('Data input (X) shape  :', X.shape)
    print('Data output (y) shape :', y.shape)
    
    # dump file
    utils.pickle_dump(X, CONFIG_FILE[f'X_{type}_path'])
    utils.pickle_dump(y, CONFIG_FILE[f'y_{type}_path'])
    
    if return_file:
        return X, y  
    

# Run files
if __name__ == '__main__':
    # Load config
    CONFIG_FILE = utils.config_load()
    
    # Start data pipeline
    read_data()
    split_train_test()
    split_input_ouput(type='train')
    split_input_ouput(type='test')
    