import numpy as np


def string_to_binary_array(orig_string):
    """Convert a string to a binary array."""
    binary_list = ['{0:08b}'.format(ord(char)) for char in orig_string]
    binary_string = ''.join(binary_list)
    binary_array = np.array([int(bit)
                            for bit in binary_string], dtype=np.uint8)

    return binary_array


def binary_array_to_string(binary_array):
    """Convert a binary array back to a string."""
    binary_string = ''.join(binary_array.astype(str))
    byte_list = [binary_string[i:i+8] for i in range(0, len(binary_string), 8)]
    chars = [chr(int(byte, 2)) for byte in byte_list]
    orig_string = ''.join(chars)

    return orig_string


def fill_end_zeros(array, num):
    """Pad an array with zeros at the end to make its length a multiple of a specified number."""
    length = len(array)
    if len(array) % num == 0:
        return array
    else:
        num_zeros = num - (length % num)
        adjusted_array = np.pad(array, (0, num_zeros),
                                mode='constant', constant_values=0)

        return adjusted_array


def write_message_to_file(message, filename):
    """Write a message string to a specified file"""
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(message)
