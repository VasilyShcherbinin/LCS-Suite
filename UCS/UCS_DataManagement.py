"""
Name:        UCS_DataManagement.py
Authors:     Ryan Urbanowicz - Written at Dartmouth College, Hanover, NH, USA
Contact:     ryan.j.urbanowicz@darmouth.edu
Created:     November 1, 2013
Description: Able to manage both training and testing data.  This module loads the dataset, detects and characterizes all attributes in the dataset, 
             handles missing data, and finally formats the data so that it may be conveniently utilized by LCS.
             
---------------------------------------------------------------------------------------------------------------------------------------------------------
LCS: Educational Learning Classifier System - A basic LCS coded for educational purposes.  This LCS algorithm uses supervised learning, and thus is most
similar to "UCS", an LCS algorithm published by Ester Bernado-Mansilla and Josep Garrell-Guiu (2003) which in turn is based heavily on "XCS", an LCS 
algorithm published by Stewart Wilson (1995).  

Copyright (C) 2013 Ryan Urbanowicz 
This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the 
Free Software Foundation; either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABLILITY 
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, 
Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
---------------------------------------------------------------------------------------------------------------------------------------------------------
"""

import random

# Import Required Modules---------------
from UCS.UCS_Constants import *


# --------------------------------------

class DataManagement:
    def __init__(self, trainFile, testFile, infoList=None):
        # Set random seed if specified.-----------------------------------------------
        if cons.useSeed:
            random.seed(cons.randomSeed)
        else:
            random.seed(None)

        # Initialize global variables-------------------------------------------------
        self.numAttributes = None  # The number of attributes in the input file.
        self.arephenotypeIDs = False  # Does the dataset contain a column of phenotype IDs? (If so, it will not be included as an attribute)
        self.phenotypeIDRef = None  # The column reference for phenotype IDs
        self.phenotypeRef = None  # The column reference for the Class/phenotype column
        self.discretephenotype = True  # Is the Class/phenotype Discrete? (False = Continuous)
        self.attributeInfo = []  # Stores Discrete (0) or Continuous (1) for each attribute
        self.phenotypeList = []  # Stores all possible discrete phenotype states/classes or maximum and minimum values for a continuous phenotype
        self.phenotypeRange = None  # Stores the difference between the maximum and minimum values for a continuous phenotype

        # Train/Test Specific-----------------------------------------------------------------------------
        self.trainHeaderList = []  # The dataset column headers for the training data
        self.testHeaderList = []  # The dataset column headers for the testing data
        self.numTrainphenotypes = None  # The number of phenotypes in the training data
        self.numTestphenotypes = None  # The number of phenotypes in the testing data

        print("----------------------------------------------------------------------------")
        print("UCS: The Complete UCS Algorithm - Niche GA + Subsumption")
        print("----------------------------------------------------------------------------")
        print("Environment: Formatting Data... ")

        # Detect Features of training data--------------------------------------------------------------------------
        rawTrainData = self.loadData(trainFile, True)  # Load the raw data.

        self.characterizeDataset(rawTrainData)  # Detect number of attributes, phenotypes, and reference locations.

        if cons.testFile == 'None':  # If no testing data is available, formatting relies solely on training data.
            data4Formating = rawTrainData
        else:
            rawTestData = self.loadData(testFile, False)  # Load the raw data.
            self.compareDataset(
                rawTestData)  # Ensure that key features are the same between training and testing datasets.
            data4Formating = rawTrainData + rawTestData  # Merge Training and Testing datasets

        self.discriminatephenotype(data4Formating)  # Determine if endpoint/phenotype is discrete or continuous.
        if self.discretephenotype:
            self.discriminateClasses(data4Formating)  # Detect number of unique phenotype identifiers.
        else:
            self.characterizephenotype(data4Formating)

        self.discriminateAttributes(data4Formating)  # Detect whether attributes are discrete or continuous.
        self.characterizeAttributes(data4Formating)  # Determine potential attribute states or ranges.

        # Format and Shuffle Datasets----------------------------------------------------------------------------------------
        if cons.testFile != 'None':
            self.testFormatted = self.formatData(
                rawTestData)  # Stores the formatted testing data set used throughout the algorithm.

        self.trainFormatted = self.formatData(
            rawTrainData)  # Stores the formatted training data set used throughout the algorithm.
        print("----------------------------------------------------------------------------")

    def loadData(self, dataFile, doTrain):
        """ Load the data file. """
        print("DataManagement: Loading Data... " + str(dataFile))
        datasetList = []
        try:
            f = open(dataFile, 'r')
        except Exception as inst:
            print(type(inst))
            print(inst.args)
            print(inst)
            print('cannot open', dataFile)
            raise
        else:
            if doTrain:
                self.trainHeaderList = f.readline().rstrip('\n').split('\t')  # strip off first row
            else:
                self.testHeaderList = f.readline().rstrip('\n').split('\t')  # strip off first row
            for line in f:
                lineList = line.strip('\n').split('\t')
                datasetList.append(lineList)
            f.close()

        return datasetList

    def characterizeDataset(self, rawTrainData):
        " Detect basic dataset parameters "
        # Detect phenotype ID's and save location if they occur.  Then save number of attributes in data.
        if cons.labelphenotypeID in self.trainHeaderList:
            self.arephenotypeIDs = True
            self.phenotypeIDRef = self.trainHeaderList.index(cons.labelphenotypeID)
            print("DataManagement: phenotype ID Column location = " + str(self.phenotypeIDRef))
            self.numAttributes = len(
                self.trainHeaderList) - 2  # one column for phenotypeID and another for the phenotype.
        else:
            self.numAttributes = len(self.trainHeaderList) - 1

        # Identify location of phenotype column
        if cons.labelphenotype in self.trainHeaderList:
            self.phenotypeRef = self.trainHeaderList.index(cons.labelphenotype)
            print("DataManagement: phenotype Column Location = " + str(self.phenotypeRef))
        else:
            print(
                "DataManagement: Error - phenotype column not found!  Check data set to ensure correct phenotype column label, or inclusion in the data.")

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

        # Store number of phenotypes in training data
        self.numTrainphenotypes = len(rawTrainData)
        print("DataManagement: Number of Attributes = " + str(self.numAttributes))
        print("DataManagement: Number of phenotypes = " + str(self.numTrainphenotypes))
        if cons.kfold == 0:
            print("DataManagement: Number of phenotypes = " + str(self.numTrainphenotypes))

    def discriminatephenotype(self, rawData):
        """ Determine whether the phenotype is Discrete(class-based) or Continuous """
        print("DataManagement: Analyzing phenotype...")
        inst = 0
        classDict = {}
        while self.discretephenotype and len(list(
                classDict.keys())) <= cons.discreteAttributeLimit and inst < self.numTrainphenotypes:  # Checks which discriminate between discrete and continuous attribute
            target = rawData[inst][self.phenotypeRef]
            if target in list(classDict.keys()):  # Check if we've seen this attribute state yet.
                classDict[target] += 1
            elif target == cons.labelMissingData:  # Ignore missing data
                print("DataManagement: Warning - Individual detected with missing phenotype information!")
                pass
            else:  # New state observed
                classDict[target] = 1
            inst += 1

        if len(list(classDict.keys())) > cons.discreteAttributeLimit:
            self.discretephenotype = False
            self.phenotypeList = [float(target), float(target)]
            print("DataManagement: phenotype Detected as Continuous.")
        else:
            print("DataManagement: phenotype Detected as Discrete.")

    def discriminateClasses(self, rawData):
        """ Determines number of classes and their identifiers. Only used if phenotype is discrete. """
        print("DataManagement: Detecting Classes...")
        inst = 0
        classCount = {}
        while inst < self.numTrainphenotypes:
            target = rawData[inst][self.phenotypeRef]
            if target in self.phenotypeList:
                classCount[target] += 1
            else:
                self.phenotypeList.append(target)
                classCount[target] = 1
            inst += 1
        print("DataManagement: Following Classes Detected:" + str(self.phenotypeList))
        for each in list(classCount.keys()):
            print("Class: " + str(each) + " count = " + str(classCount[each]))

    def compareDataset(self, rawTestData):
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

        # Stores the number of phenotypes in the testing data.
        self.numTestphenotypes = len(rawTestData)
        print("DataManagement: Number of Attributes = " + str(self.numAttributes))
        print("DataManagement: Number of phenotypes = " + str(self.numTestphenotypes))

    def discriminateAttributes(self, rawData):
        """ Determine whether attributes in dataset are discrete or continuous and saves this information. """
        print("DataManagement: Detecting Attributes...")
        self.discreteCount = 0
        self.continuousCount = 0
        for att in range(len(rawData[0])):
            if att != self.phenotypeIDRef and att != self.phenotypeRef:  # Get just the attribute columns (ignores phenotype and phenotypeID columns)
                attIsDiscrete = True
                inst = 0
                stateDict = {}
                while attIsDiscrete and len(list(
                        stateDict.keys())) <= cons.discreteAttributeLimit and inst < self.numTrainphenotypes:  # Checks which discriminate between discrete and continuous attribute
                    target = rawData[inst][att]
                    if target in list(stateDict.keys()):  # Check if we've seen this attribute state yet.
                        stateDict[target] += 1
                    elif target == cons.labelMissingData:  # Ignore missing data
                        pass
                    else:  # New state observed
                        stateDict[target] = 1
                    inst += 1

                if len(list(stateDict.keys())) > cons.discreteAttributeLimit:
                    attIsDiscrete = False
                if attIsDiscrete:
                    self.attributeInfo.append([0, []])
                    self.discreteCount += 1
                else:
                    self.attributeInfo.append([1, [float(target), float(target)]])  # [min,max]
                    self.continuousCount += 1
        print("DataManagement: Identified " + str(self.discreteCount) + " discrete and " + str(
            self.continuousCount) + " continuous attributes.")  # Debug

    def characterizeAttributes(self, rawData):
        """ Determine range (if continuous) or states (if discrete) for each attribute and saves this information"""
        print("DataManagement: Characterizing Attributes...")
        attributeID = 0
        for att in range(len(rawData[0])):
            if att != self.phenotypeIDRef and att != self.phenotypeRef:  # Get just the attribute columns (ignores phenotype and phenotypeID columns)
                for inst in range(len(rawData)):
                    target = rawData[inst][att]
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

    def characterizephenotype(self, rawData):
        """ Determine range of phenotype values. """
        print("DataManagement: Characterizing phenotype...")
        for inst in range(len(rawData)):
            target = rawData[inst][self.phenotypeRef]

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

    def formatData(self, rawData):
        """ Get the data into a format convenient for the algorithm to interact with. Specifically each phenotype is stored in a list as follows; [Attribute States, phenotype, phenotypeID] """
        formatted = []
        # Initialize data format---------------------------------------------------------
        for i in range(len(rawData)):
            formatted.append([None, None, None])  # [Attribute States, phenotype, phenotypeID]

        for inst in range(len(rawData)):
            stateList = []
            attributeID = 0
            for att in range(len(rawData[0])):
                if att != self.phenotypeIDRef and att != self.phenotypeRef:  # Get just the attribute columns (ignores phenotype and phenotypeID columns)
                    target = rawData[inst][att]

                    if self.attributeInfo[attributeID][0]:  # If the attribute is continuous
                        if target == cons.labelMissingData:
                            stateList.append(target)  # Missing data saved as text label
                        else:
                            stateList.append(float(target))  # Save continuous data as floats.
                    else:  # If the attribute is discrete - Format the data to correspond to the GABIL (DeJong 1991)
                        stateList.append(
                            target)  # missing data, and discrete variables, all stored as string objects
                    attributeID += 1

            # Final Format-----------------------------------------------
            formatted[inst][0] = stateList  # Attribute states stored here
            if self.discretephenotype:
                formatted[inst][1] = rawData[inst][self.phenotypeRef]  # phenotype stored here
            else:
                formatted[inst][1] = float(rawData[inst][self.phenotypeRef])
            if self.arephenotypeIDs:
                formatted[inst][2] = rawData[inst][self.phenotypeIDRef]  # phenotype ID stored here
            else:
                pass  # phenotype ID neither given nor required.
            # -----------------------------------------------------------
        random.shuffle(
            formatted)  # One time randomization of the order the of the phenotypes in the data, so that if the data was ordered by phenotype, this potential learning bias (based on phenotype ordering) is eliminated.
        return formatted

    def splitDataIntoKSets(self):
        """ divide data set into kfold sets. """
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
        self.testFormatted = self.folds[fold_id]
        self.numTrainphenotypes = len(self.trainFormatted)
        self.numTestphenotypes = len(self.testFormatted)
        print("DataManagement: Number of Train Instances = " + str(self.numTrainphenotypes))
        print("DataManagement: Number of Test Instances = " + str(self.numTestphenotypes))
