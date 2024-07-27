import numpy as np


def file_to_binary_1D_arr(file_path):
    """ Converts the contents of a file to a 1D numpy array of binary values (0s and 1s)."""
    with open(file_path, 'rb') as file:
        binary_data = file.read()
        binary_array = np.array([int(bit) for byte in binary_data for bit in f"{byte:08b}"], dtype=np.uint8)
        print(f"[INFO] The file '{file_path}' has been successfully converted to a 1D numpy array.")
    return binary_array


def binary_1D_arr_to_file(binary_array, file_path):
    """Converts a 1D numpy array of binary values back to a file."""
    binary_string = ''.join(binary_array.astype(str))
    byte_list = [int(binary_string[i:i+8], 2) for i in range(0, len(binary_string), 8)]
    byte_arr = bytearray(byte_list)
    
    with open(file_path, 'wb') as file:
        file.write(byte_arr)
        print(f"[INFO] The binary array has been successfully converted to the file '{file_path}'.")
      
      


def string_to_binary_array(orig_string):
    binary_list = ['{0:08b}'.format(ord(char)) for char in orig_string]
    binary_string = ''.join(binary_list)
    binary_array = np.array([int(bit) for bit in binary_string], dtype=np.uint8)
    
    return binary_array


def binary_array_to_string(binary_array):
    binary_string = ''.join(binary_array.astype(str))
    byte_list = [binary_string[i:i+8] for i in range(0, len(binary_string), 8)]
    chars = [chr(int(byte, 2)) for byte in byte_list]
    orig_string = ''.join(chars)
    
    return orig_string

###########
def random_xor_key(n):
    rand = np.random.randint(2, size=n)
    return rand

###########
def add_EOT_sequence(message):
    sequence = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    new_message = np.concatenate((message, sequence))
    return new_message


def check_EOT_sequence(message):
    sequence = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    return np.array_equal(message[-48:], sequence)

def fill_end_zeros(array, num):
    length = len(array)
    if len(array) % num == 0:
        return array
    else:
        num_zeros = num - (length % num)
        adjusted_array = np.pad(array, (0, num_zeros), mode='constant', constant_values=0)

        return adjusted_array