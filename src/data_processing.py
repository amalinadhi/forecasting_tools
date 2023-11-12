import pandas as pd
import utils


# Import functions
def preprocess_data(
    data=None, 
    type=None, 
    CONFIG_FILE=None, 
    return_file=False):
    """preprocess the data"""
    # load the data
    if type is not None:
        fname = CONFIG_FILE[f'X_{type}_path']
        data = utils.pickle_load(fname)
        
        # Validate
        print(f'Data on {fname} has been loaded')
        print('Data shape :', data.shape)
        
    # Extract date
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    
    # Validate
    print('Data shape:', data.shape)
    
    # Dump data
    if type is not None:
        utils.pickle_dump(data, CONFIG_FILE[f'X_{type}_clean_path'])
    
    if return_file:
        return data
    
    
# Run files
if __name__ == '__main__':
    # Load config
    CONFIG_FILE = utils.config_load()
    
    # Start preprocess data
    preprocess_data(type='train', CONFIG_FILE=CONFIG_FILE)
    preprocess_data(type='test', CONFIG_FILE=CONFIG_FILE)