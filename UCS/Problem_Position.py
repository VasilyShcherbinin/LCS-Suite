"""
Name: Problem_Position.py
Author: Vasily Shcherbinin
Created: December 30, 2018
Description: Position Problem
Input: Binary string
Output: Position of the left-most one-valued bit.
"""

import random

if __name__ == '__main__':

    def generate_position_data(myfile, bits, instances):
        """ """
        print("Problem_Position: Attempting to Generate position dataset with " + str(instances) + " instances.")
        num_bits = sanity_check(bits)

        if num_bits is None:
            print("Problem_Position: ERROR - Specified binary string bits is smaller than or 0")
        else:
            fp = open("Demo_Datasets/" + myfile, "w")
            # Make File Header
            for i in range(num_bits):
                fp.write('B_' + str(i) + "\t")  # Bits
            fp.write("Class" + "\n")  # Class

            for i in range(instances):
                instance = generate_position_instance(bits)
                for j in instance[0]:
                    fp.write(str(j) + "\t")
                fp.write(str(instance[1]) + "\n")

            fp.close()
            print("Problem_Position: File Generated")


    def generate_position_instance(bits):
        """ """
        num_bits = sanity_check(bits)
        if num_bits is None:
            print("Problem_Position: ERROR - Specified binary string size is smaller than or 0")
        else:
            condition = []
            position = 0

            # Generate random boolean string
            for i in range(bits):
                condition.append(str(random.randint(0, 1)))

            for j in range(len(condition)):
                if int(condition[j]) == 1:
                    position = j
                    break

            output = position

            return [condition, output]


    def generate_complete_position_data(myfile, bits):
        """ Attempts to generate a complete non-redundant position dataset."""

        print("Problem_Position: Attempting to generate complete position dataset")
        num_bits = sanity_check(bits)

        if num_bits is None:
            print("Problem_Position: ERROR - Specified binary string bits is smaller than or 0")
        else:
            try:
                fp = open("Demo_Datasets/"+myfile, "w")
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

                    bin_list = list(binary)
                    output = 0

                    for j in range(len(bin_list)):
                        if int(bin_list[j]) == 1:
                            output = j
                            break

                    for j in range(num_bits):
                        fp.write(binary[j] + "\t")

                    fp.write(str(output) + "\n")

                fp.close()
                print("Problem_Position: Dataset Generation Complete")

            except:
                print(
                    "ERROR - Cannot generate all data instances for specified binary due to computational limitations")


    def sanity_check(bits):
        if bits > 0:
            return bits
        return None

    bits = 8
    instances = 10

    # generate_position_data(str(bits)+"-"+str(instances)+"Position_Data.txt", bits, instances)
    generate_complete_position_data(str(bits) + "Position_Data_Complete.txt", bits)