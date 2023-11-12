import yaml
import joblib


CONFIG_DIR = 'config/config.yaml'


def config_load():
    """Load the config files"""
    try:
        with open(CONFIG_DIR, 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError as error:
        raise RuntimeError('Config file not found in path')
    
    return config

def pickle_load(file_path):
    """Load pickle file"""
    return joblib.load(file_path)

def pickle_dump(data, file_path):
    """Dump data to a pickle file"""
    joblib.dump(data, file_path)
    print(f'Data has been dumped to path: {file_path}')