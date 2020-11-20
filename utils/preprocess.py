import numpy as np

def window_offset(x, window_size):
    '''
    Note this assume next step label prediction with a stride of 1. Also assumes we don't use current window. 
    Args
    x (numpy array): Input time series
    window_size (int): Sliding window size. 
    
    Return
        [(n+1) x window_size] NumPy array 
    '''
    pad_zeros = np.zeros(window_size)
    x_pad = np.concatenate([pad_zeros, x])
    windows = [x_pad[i:i+window_size] for i in range(len(x_pad)-(window_size))]
    return np.array(windows)

def window(x, window_size):
    '''
    Note this assume next step label prediction with a stride of 1. 
    Args
    x (numpy array): Input time series
    window_size (int): Sliding window size. 
    
    Return
        [(n+1) x window_size] NumPy array 
    '''
    pad_zeros = np.zeros(window_size-1)
    x_pad = np.concatenate([pad_zeros, x])
    windows = [x_pad[i:i+window_size] for i in range(len(x_pad)-(window_size-1))]
    return np.array(windows)

