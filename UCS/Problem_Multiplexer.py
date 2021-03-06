
import random

bits = 11
instances = 10


def randomize():
    with open("../UCS/Demo_Datasets/" + str(bits) + "Multiplexer_Data_Complete.txt", 'r') as source:
        data = [(random.random(), line) for line in source]
    data[1:] = sorted(data[1:])
    with open("../UCS/Demo_Datasets/" + str(bits) + "Multiplexer_Data_Complete_Randomized.txt", 'w') as target:
        for _, line in data:
            target.write(line)


def generate_multiplexer_data(myfile, num_bits, instances):
    """ """
    print("Problem_Multiplexer: Generate multiplexer dataset with " + str(instances) + " instances.")
    first = solve_equation(num_bits)
    if first is None:
        print("Problem_Multiplexer: ERROR - The multiplexer takes # of bits as 3,6,11,20,37,70,135,264")

    else:
        fp = open("Demo_Datasets/" + myfile, "w")
        # Make File Header
        for i in range(first):
            fp.write('A_' + str(i) + "\t")  # Address Bits

        for i in range(num_bits - first):
            fp.write('R_' + str(i) + "\t")  # Register Bits
        fp.write("Class" + "\n")  # State found at Register Bit

        for i in range(instances):
            instance = generate_multiplexer_instance(num_bits)
            for j in instance[0]:
                fp.write(str(j) + "\t")
            fp.write(str(instance[1]) + "\n")


def generate_multiplexer_instance(num_bits):
    """ """
    first = solve_equation(num_bits)
    if first is None:
        print("Problem_Multiplexer: ERROR - The multiplexer takes # of bits as 3,6,11,20,37,70,135,264")

    else:
        condition = []
        # Generate random boolean string
        for i in range(num_bits):
            condition.append(str(random.randint(0, 1)))

        gates = ""

        for j in range(first):
            gates += condition[j]

        gates_decimal = int(gates, 2)
        output = condition[first + gates_decimal]

        return [condition, output]


def generate_complete_multiplexer_data(myfile, num_bits):
    """ Attempts to generate a complete non-redundant multiplexer dataset.  Ability to generate the entire dataset is computationally limited.
    We had success generating up to the complete 20-multiplexer dataset"""

    print("Problem_Multiplexer: Attempting to generate multiplexer dataset")
    first = solve_equation(num_bits)

    if first is None:
        print("Problem_Multiplexer: ERROR - The multiplexer takes # of bits as 3,6,11,20,37,70,135,264")
    else:
        try:
            fp = open("Demo_Datasets/" + myfile, "w")
            # Make File Header
            for i in range(first):
                fp.write('A_' + str(i) + "\t")  # Address Bits

            for i in range(num_bits - first):
                fp.write('R_' + str(i) + "\t")  # Register Bits
            fp.write("Class" + "\n")  # State found at Register Bit

            for i in range(2 ** num_bits):
                binary_str = bin(i)
                string_array = binary_str.split('b')
                binary = string_array[1]

                while len(binary) < num_bits:
                    binary = "0" + binary

                gates = ""
                for j in range(first):
                    gates += binary[j]

                gates_decimal = int(gates, 2)
                output = binary[first + gates_decimal]

                # fp.write(str(i)+"\t")
                for j in binary:
                    fp.write(j + "\t")
                fp.write(output + "\n")

            fp.close()
            print("Problem_Multiplexer: Dataset Generation Complete")

        except:
            print(
                "ERROR - Cannot generate all data instances for specified multiplexer due to computational limitations")


def solve_equation(num_bits):
    for i in range(1000):
        if i + 2 ** i == num_bits:
            return i
    return None


if __name__ == '__main__':

    generate_complete_multiplexer_data(str(bits) + "Multiplexer_Data_Complete.txt", bits)  # 3,6,11,20,37
    randomize()
    # generate_multiplexer_data("Multiplexer_Data.txt", bits, instances)
