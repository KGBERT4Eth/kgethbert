import pickle

def save_pkl(data, file_path):
    """
    pkl
    :param data: 
    :param file_path: pkl
    """
    with open(file_path, 'wb') as pkl_file:
        pickle.dump(data, pkl_file)

def load_pkl(file_path):
    """
    pkl
    :param file_path: pkl
    :return: 
    """
    with open(file_path, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data