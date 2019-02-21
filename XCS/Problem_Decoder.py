"""
Name: Problem_Decoder.py
Author: Vasily Shcherbinin
Created: December 30, 2018
Description: Decoder Problem
Input: Binary string
Output: Decimal value of the binary string
"""

import random

bits = 11
instances = 10


def randomize():
    with open("../XCS/Demo_Datasets/" + str(bits) + "Decoder_Data_Complete.txt", 'r') as source:
        data = [(random.random(), line) for line in source]
    data[1:] = sorted(data[1:])
    with open("../XCS/Demo_Datasets/" + str(bits) + "Decoder_Data_Complete_Randomized.txt", 'w') as target:
        for _, line in data:
            target.write(line)


def generate_decoder_data(myfile, bits, instances):
    """ """
    print("Problem_Decoder: Attempting to Generate decoder dataset with " + str(instances) + " instances.")
    num_bits = sanity_check(bits)

    if num_bits is None:
        print("Problem_Decoder: ERROR - Specified binary string bits is smaller than or 0")
    else:
        fp = open("Demo_Datasets/" + myfile, "w")
        # Make File Header
        for i in range(num_bits):
            fp.write('B_' + str(i) + "\t")  # Bits
        fp.write("Class" + "\n")  # Class

        for i in range(instances):
            instance = generate_decoder_instance(bits)
            for j in instance[0]:
                fp.write(str(j) + "\t")
            fp.write(str(instance[1]) + "\n")

        fp.close()
        print("Problem_Decoder: File Generated")


def generate_decoder_instance(bits):
    """ """
    num_bits = sanity_check(bits)
    if num_bits is None:
        print("Problem_Decoder: ERROR - Specified binary string size is smaller than or 0")
    else:
        condition = ""

        # Generate random boolean string
        for i in range(bits):
            condition = condition + (str(random.randint(0, 1)))

        output = int(condition, base=2)

        return [condition, output]


def generate_complete_decoder_data(myfile, bits):
    """ Attempts to generate a complete non-redundant decoder dataset."""

    print("Problem_Decoder: Attempting to generate complete decoder dataset")
    num_bits = sanity_check(bits)

    if num_bits is None:
        print("Problem_Decoder: ERROR - Specified binary string bits is smaller than or 0")
    else:
        try:
            fp = open("Demo_Datasets/" + myfile, "w")
            # Make File Header
            for i in range(num_bits):
                fp.write('B_' + str(i) + "\t")  # Bits
            fp.write("Class" + "\n")  # Class

            for i in range(2 ** num_bits):
                binary_str = bin(i)
                string_array = binary_str.split('b')
                binary = string_array[1]

                while len(binary) < num_bits:
                    binary = "0" + binary

                output = int(binary, base=2)

                for j in range(num_bits):
                    fp.write(binary[j] + "\t")

                fp.write(str(output) + "\n")

            fp.close()
            print("Problem_Decoder: Dataset Generation Complete")

        except:
            print(
                "ERROR - Cannot generate all data instances for specified binary due to computational limitations")


def sanity_check(input_bits):
    if input_bits > 0:
        return input_bits
    return None


if __name__ == '__main__':
    # generate_decoder_data(str(bits)+"-"+str(instances)+"Decoder_Data.txt", bits, instances)
    generate_complete_decoder_data(str(bits) + "Decoder_Data_Complete.txt", bits)
    randomize()
