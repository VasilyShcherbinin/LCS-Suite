"""
Name: Problem_Parity.py
Author: Vasily Shcherbinin
Created: December 29, 2018
Description: Parity Problem
Input: Binary string
Output: No. of 1’s modulo 2
"""

import random

bits = 11
instances = 10


def generate_parity_instance(bits):
    """ """
    num_bits = sanity_check(bits)
    if num_bits is None:
        print("Problem_Parity: ERROR - Specified binary string size is smaller than or 0")
    else:
        condition = []
        one_count = 0

        # Generate random boolean string
        for i in range(bits):
            condition.append(str(random.randint(0, 1)))

        for j in range(len(condition)):
            if int(condition[j]) == 1:
                one_count += 1

        if one_count % 2 == 0:
            output = 0
        else:
            output = 1

        return [condition, output]


def generate_complete_parity_data(myfile, bits):
    """ Attempts to generate a complete non-redundant parity dataset."""

    print("Problem_Parity: Attempting to generate complete parity dataset")
    num_bits = sanity_check(bits)

    if num_bits is None:
        print("Problem_Parity: ERROR - Specified binary string bits is smaller than or 0")
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

                one_count = 0

                for j in binary:
                    if int(j) == 1:
                        one_count += 1

                if one_count % 2 == 0:
                    output = 0
                else:
                    output = 1

                for j in range(num_bits):
                    fp.write(binary[j] + "\t")

                fp.write(str(output) + "\n")

            fp.close()
            print("Problem_Parity: Dataset Generation Complete")

        except:
            print(
                "ERROR - Cannot generate all data instances for specified binary due to computational limitations")


def sanity_check(bits):
    if bits > 0:
        return bits
    return None


def generate_parity_data(myfile, bits, instances):
    """ """
    print("Problem_Parity: Attempting to Generate parity dataset with " + str(instances) + " instances.")
    num_bits = sanity_check(bits)

    if num_bits is None:
        print("Problem_Parity: ERROR - Specified binary string bits is smaller than or 0")
    else:
        fp = open("Demo_Datasets/" + myfile, "w")
        # Make File Header
        for i in range(num_bits):
            fp.write('B_' + str(i) + "\t")  # Bits
        fp.write("Class" + "\n")  # Class

        for i in range(instances):
            instance = generate_parity_instance(bits)
            for j in instance[0]:
                fp.write(str(j) + "\t")
            fp.write(str(instance[1]) + "\n")

        fp.close()
        print("Problem_Parity: File Generated")


def randomize():
    with open("../XCS/Demo_Datasets/"+str(bits)+"Parity_Data_Complete.txt", 'r') as source:
        data = [(random.random(), line) for line in source]
    data[1:] = sorted(data[1:])
    with open("../XCS/Demo_Datasets/"+str(bits)+"Parity_Data_Complete_Randomized.txt", 'w') as target:
        for _, line in data:
            target.write(line)


if __name__ == '__main__':

    #generate_parity_data(str(bits)+"-"+str(instances)+"Parity_Data.txt", bits, instances)
    generate_complete_parity_data(str(bits)+"Parity_Data_Complete.txt", bits)
    randomize()