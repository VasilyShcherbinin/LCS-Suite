# Import Required Modules---------------
import random

from XCS.XCS_Constants import cons


# --------------------------------------

class DataManagement:

    def __init__(self, trainFile, testFile):
        # Initialize global variables-------------------------------------------------
        self.numAttributes = None  # The number of attributes in the input file.
        self.arephenotypeIDs = False  # Does the dataset contain a column of Instance IDs? (If so, it will not be included as an attribute)
        self.phenotypeIDRef = None  # The column reference for Instance IDs
        self.phenotypeRef = None  # The column reference for the Class/Phenotype column
        self.discretephenotype = True  # Is the Class/Phenotype Discrete? (False = Continuous)
        self.attributeInfo = []  # Stores Discrete (0) or Continuous (1) for each attribute
        self.phenotypeList = []  # Stores all possible discrete phenotype states/classes or maximum and minimum values for a continuous phenotype
        self.phenotypeRange = None  # Stores the difference between the maximum and minimum values for a continuous phenotype

        # Train/Test Specific-----------------------------------------------------------------------------
        self.trainHeaderList = []  # The dataset column headers for the training data
        self.testHeaderList = []  # The dataset column headers for the testing data
        self.numTrainphenotypes = None  # The number of instances in the training data
        self.numTestphenotypes = None  # The number of instances in the testing data

        print("----------------------------------------------------------------------------")
        print("XCS Code Demo:")
        print("----------------------------------------------------------------------------")
        print("Environment: Formatting Data... ")

        # Detect Features of training data--------------------------------------------------------------------------
        raw_train_data = self.loadData(trainFile, True)  # Load the raw data.

        self.characterizeDataset(raw_train_data)  # Detect number of attributes, instances, and reference locations.

        if cons.testFile == 'None':  # If no testing data is available, formatting relies solely on training data.
            tobe_formatted = raw_train_data
        else:
            raw_test_data = self.loadData(testFile, False)  # Load the raw data.
            self.compareDataset(
                raw_test_data)  # Ensure that key features are the same between training and testing datasets.
            tobe_formatted = raw_train_data + raw_test_data  # Merge Training and Testing datasets

        self.discriminatePhenotype(tobe_formatted)  # Determine if endpoint/phenotype is discrete or continuous.
        if self.discretephenotype:
            self.discriminateClasses(tobe_formatted)  # Detect number of unique phenotype identifiers.
        else:
            self.characterizePhenotype(tobe_formatted)

        self.discriminateAttributes(tobe_formatted)  # Detect whether attributes are discrete or continuous.
        self.characterizeAttributes(tobe_formatted)  # Determine potential attribute states or ranges.

        # Format and Shuffle Datasets----------------------------------------------------------------------------------------
        if cons.testFile != 'None':
            self.formatted_test_data = self.formatData(
                raw_test_data)  # Stores the formatted testing data set used throughout the algorithm.

        self.trainFormatted = self.formatData(
            raw_train_data)  # Stores the formatted training data set used throughout the algorithm.
        print("----------------------------------------------------------------------------")

    def loadData(self, dat_file, do_train):
        """ Load the data file. """
        print("DataManagement: Loading Data... " + str(dat_file))
        dataset_list = []
        try:
            f = open(dat_file, 'r')
        except Exception as inst:
            print(type(inst))
            print(inst.args)
            print(inst)
            print('cannot open', dat_file)
            raise
        else:
            if do_train:
                self.trainHeaderList = f.readline().rstrip('\n').split('\t')  # strip off first row
            else:
                self.testHeaderList = f.readline().rstrip('\n').split('\t')  # strip off first row
            for line in f:
                line_list = line.strip('\n').split('\t')
                dataset_list.append(line_list)
            f.close()

        return dataset_list

    def characterizeDataset(self, raw_train_data):
        " Detect basic dataset parameters "
        # Detect Instance ID's and save location if they occur.  Then save number of attributes in data.
        if cons.labelphenotypeID in self.trainHeaderList:
            self.arephenotypeIDs = True
            self.phenotypeIDRef = self.trainHeaderList.index(cons.labelphenotypeID)
            print("DataManagement: Instance ID Column location = " + str(self.phenotypeIDRef))
            self.numAttributes = len(
                self.trainHeaderList) - 2  # one column for InstanceID and another for the phenotype.
        else:
            self.numAttributes = len(self.trainHeaderList) - 1

        # Identify location of phenotype column
        if cons.labelphenotype in self.trainHeaderList:
            self.phenotypeRef = self.trainHeaderList.index(cons.labelphenotype)
            print("DataManagement: Phenotype Column Location = " + str(self.phenotypeRef))
        else:
            print(
                "DataManagement: Error - Phenotype column not found!  Check data set to ensure correct phenotype column label, or inclusion in the data.")

        # Adjust training header list to just include attributes labels
        if self.arephenotypeIDs:
            if self.phenotypeRef > self.phenotypeIDRef:
                self.trainHeaderList.pop(self.phenotypeRef)
                self.trainHeaderList.pop(self.phenotypeIDRef)
            else:
                self.trainHeaderList.pop(self.phenotypeIDRef)
                self.trainHeaderList.pop(self.phenotypeRef)
        else:
            self.trainHeaderList.pop(self.phenotypeRef)

        # Store number of instances in training data
        print("DataManagement: Number of Attributes = " + str(self.numAttributes))
        self.numTrainphenotypes = len(raw_train_data)
        if cons.kfold == 0:
            print("DataManagement: Number of Instances = " + str(self.numTrainphenotypes))

    def discriminatePhenotype(self, raw_data):
        """ Determine whether the phenotype is Discrete(class-based) or Continuous """
        print("DataManagement: Analyzing Phenotype...")
        inst = 0
        class_dict = {}
        while self.discretephenotype and len(list(
                class_dict.keys())) <= cons.discreteAttributeLimit and inst < self.numTrainphenotypes:  # Checks which discriminate between discrete and continuous attribute
            target = raw_data[inst][self.phenotypeRef]
            if target in list(class_dict.keys()):  # Check if we've seen this attribute state yet.
                class_dict[target] += 1
            elif target == cons.labelMissingData:  # Ignore missing data
                print("DataManagement: Warning - Individual detected with missing phenotype information!")
                pass
            else:  # New state observed
                class_dict[target] = 1
            inst += 1

        if len(list(class_dict.keys())) > cons.discreteAttributeLimit:
            self.discretephenotype = False
            self.phenotypeList = [float(target), float(target)]
            print("DataManagement: Phenotype Detected as Continuous.")
        else:
            print("DataManagement: Phenotype Detected as Discrete.")

    def discriminateClasses(self, raw_data):
        """ Determines number of classes and their identifiers. Only used if phenotype is discrete. """
        print("DataManagement: Detecting Classes...")
        inst = 0
        class_count = {}
        while inst < self.numTrainphenotypes:
            target = raw_data[inst][self.phenotypeRef]
            if int(target) in self.phenotypeList:
                class_count[target] += 1
            else:
                self.phenotypeList.append(int(target))
                class_count[target] = 1
            inst += 1
        print("DataManagement: Following Classes Detected:" + str(self.phenotypeList))
        for each in list(class_count.keys()):
            print("Class: " + str(each) + " count = " + str(class_count[each]))

    def compareDataset(self, raw_test_data):
        " Ensures that the attributes in the testing data match those in the training data.  Also stores some information about the testing data. "
        if self.arephenotypeIDs:
            if self.phenotypeRef > self.phenotypeIDRef:
                self.testHeaderList.pop(self.phenotypeRef)
                self.testHeaderList.pop(self.phenotypeIDRef)
            else:
                self.testHeaderList.pop(self.phenotypeIDRef)
                self.testHeaderList.pop(self.phenotypeRef)
        else:
            self.testHeaderList.pop(self.phenotypeRef)

        if self.trainHeaderList != self.testHeaderList:
            print("DataManagement: Error - Training and Testing Dataset Headers are not equivalent")

        # Stores the number of instances in the testing data.
        self.numTestphenotypes = len(raw_test_data)
        print("DataManagement: Number of Attributes = " + str(self.numAttributes))
        print("DataManagement: Number of Instances = " + str(self.numTestphenotypes))

    def discriminateAttributes(self, raw_data):
        """ Determine whether attributes in dataset are discrete or continuous and saves this information. """
        print("DataManagement: Detecting Attributes...")
        self.discrete_count = 0
        self.continuous_count = 0
        for att in range(len(raw_data[0])):
            if att != self.phenotypeIDRef and att != self.phenotypeRef:  # Get just the attribute columns (ignores phenotype and instanceID columns)
                is_att_discrete = True
                inst = 0
                state_dict = {}
                while is_att_discrete and len(list(
                        state_dict.keys())) <= cons.discreteAttributeLimit and inst < self.numTrainphenotypes:  # Checks which discriminate between discrete and continuous attribute
                    target = raw_data[inst][att]
                    if target in list(state_dict.keys()):  # Check if we've seen this attribute state yet.
                        state_dict[target] += 1
                    elif target == cons.labelMissingData:  # Ignore missing data
                        pass
                    else:  # New state observed
                        state_dict[target] = 1
                    inst += 1

                if len(list(state_dict.keys())) > cons.discreteAttributeLimit:
                    is_att_discrete = False
                if is_att_discrete:
                    self.attributeInfo.append([0, []])
                    self.discrete_count += 1
                else:
                    self.attributeInfo.append([1, [float(target), float(target)]])  # [min,max]
                    self.continuous_count += 1
        print("DataManagement: Identified " + str(self.discrete_count) + " discrete and " + str(
            self.continuous_count) + " continuous attributes.")  # Debug

    def characterizeAttributes(self, raw_data):
        """ Determine range (if continuous) or states (if discrete) for each attribute and saves this information"""
        print("DataManagement: Characterizing Attributes...")
        attributeID = 0
        for att in range(len(raw_data[0])):
            if att != self.phenotypeIDRef and att != self.phenotypeRef:  # Get just the attribute columns (ignores phenotype and instanceID columns)
                for inst in range(len(raw_data)):
                    target = raw_data[inst][att]
                    if not self.attributeInfo[attributeID][0]:  # If attribute is discrete
                        if target in self.attributeInfo[attributeID][1] or target == cons.labelMissingData:
                            pass  # NOTE: Could potentially store state frequency information to guide learning.
                        else:
                            self.attributeInfo[attributeID][1].append(target)
                    else:  # If attribute is continuous

                        # Find Minimum and Maximum values for the continuous attribute so we know the range.
                        if target == cons.labelMissingData:
                            pass
                        elif float(target) > self.attributeInfo[attributeID][1][1]:  # error
                            self.attributeInfo[attributeID][1][1] = float(target)
                        elif float(target) < self.attributeInfo[attributeID][1][0]:
                            self.attributeInfo[attributeID][1][0] = float(target)
                        else:
                            pass
                attributeID += 1

    def characterizePhenotype(self, raw_data):
        """ Determine range of phenotype values. """
        print("DataManagement: Characterizing Phenotype...")
        for inst in range(len(raw_data)):
            target = raw_data[inst][self.phenotypeRef]

            # Find Minimum and Maximum values for the continuous phenotype so we know the range.
            if target == cons.labelMissingData:
                pass
            elif float(target) > self.phenotypeList[1]:
                self.phenotypeList[1] = float(target)
            elif float(target) < self.phenotypeList[0]:
                self.phenotypeList[0] = float(target)
            else:
                pass
        self.phenotypeRange = self.phenotypeList[1] - self.phenotypeList[0]

    def formatData(self, raw_data):
        """ Get the data into a format convenient for the algorithm to interact with. Specifically each instance is stored in a list as follows; [Attribute States, Phenotype, InstanceID] """
        formatted = []
        # Initialize data format---------------------------------------------------------
        for _ in range(len(raw_data)):
            formatted.append([None, None, None])  # [Attribute States, Phenotype, InstanceID]

        for inst in range(len(raw_data)):
            state_list = [0] * self.numAttributes
            attributeID = 0
            for att in range(len(raw_data[0])):
                if att != self.phenotypeIDRef and att != self.phenotypeRef:  # Get just the attribute columns (ignores phenotype and instanceID columns)
                    target = raw_data[inst][att]
                    # If the attribute is discrete - Format the data to correspond to the GABIL (DeJong 1991)
                    if target == cons.labelMissingData:
                        state_list[attributeID] = target
                    else:
                        state_list[attributeID] = int(
                            target)  # missing data, and discrete variables, all stored as string objects
                attributeID += 1

            # Final Format-----------------------------------------------
            formatted[inst][0] = state_list  # Attribute states stored here
            if self.discretephenotype:
                formatted[inst][1] = int(raw_data[inst][self.phenotypeRef])  # phenotype stored here
            else:
                formatted[inst][1] = float(raw_data[inst][self.phenotypeRef])
            if self.arephenotypeIDs:
                formatted[inst][2] = int(raw_data[inst][self.phenotypeIDRef])  # Instance ID stored here
            else:
                pass  # instance ID neither given nor required.
            # -----------------------------------------------------------
        # random.shuffle(formatted) #One time randomization of the order the of the instances in the data, so that if the data was ordered by phenotype, this potential learning bias (based on instance ordering) is eliminated.
        return formatted

    def splitFolds(self):
        """ divide data set into kfold sets. """
        data_size = len(self.trainFormatted)
        class_counts = [0] * len(self.phenotypeList)
        for instance in self.trainFormatted:
            class_counts[self.phenotypeList.index(instance[1])] += 1
        fold_size = int(data_size / cons.kfold)
        split_again = True
        while split_again:
            split_again = False
            self.folds = [[] for _ in range(cons.kfold)]
            start_point = 0
            for i in range(cons.kfold):
                end_point = start_point + fold_size
                if i < data_size % cons.kfold:
                    end_point += 1
                self.folds[i] = self.trainFormatted[start_point:end_point]
                start_point = end_point
                fold_class_counts = [0] * len(self.phenotypeList)
                for instance in self.folds[i]:
                    fold_class_counts[self.phenotypeList.index(instance[1])] += 1
                for j in range(len(self.phenotypeList)):
                    if fold_class_counts[j] == class_counts[j]:
                        random.shuffle(self.trainFormatted)
                        split_again = True

    def splitFolds2(self):
        """ divide data set into kfold sets. """
        self.trainFormatted = stratify(self.trainFormatted)
        data_size = len(self.trainFormatted)
        self.folds = [[] for _ in range(cons.kfold)]
        for fold_id in range(cons.kfold):
            fold_size = int(data_size / cons.kfold)
            if fold_id < data_size % cons.kfold:
                fold_size += 1
                offset = fold_id
            else:
                offset = data_size % cons.kfold
            first = fold_id * (int(data_size / cons.kfold)) + offset
            self.folds[fold_id] = self.trainFormatted[first: (first + fold_size)]

    def selectTrainTestSets(self, fold_id):
        """ select subset for testing and the rest for training. """
        self.trainFormatted = []
        for i in range(len(self.folds)):
            if i != fold_id:
                self.trainFormatted += self.folds[i]
        self.formatted_test_data = self.folds[fold_id]
        self.numTrainphenotypes = len(self.trainFormatted)
        self.numTestphenotypes = len(self.formatted_test_data)
        print("DataManagement: Number of Instances = " + str(self.numTrainphenotypes))
        print("DataManagement: Number of Instances = " + str(self.numTestphenotypes))


def stratify(all_data):
    """ divide data set into kfold sets. """
    # sort by class
    index = 1
    numb_instances = len(all_data)
    while index < numb_instances:
        instance1 = all_data[index - 1]
        for j in range(index, numb_instances):
            instance2 = all_data[j]
            if instance1[1] == instance2[1]:
                # swap(index, j)
                temp = all_data[index]
                all_data[index] = all_data[j]
                all_data[j] = temp
                index += 1
        index += 1
    # rearrange classes to kfold trunks.
    stratified_data = []
    start = 0
    while len(stratified_data) < numb_instances:
        j = start
        while j < numb_instances:
            stratified_data.append(all_data[j])
            j += cons.kfold
        start += 1
    return stratified_data
