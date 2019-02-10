# Import Required Modules--------------------------------------
from XCS.XCS_Constants import *
from XCS.XCS_DataManagement import DataManagement


# -------------------------------------------------------------

class Offline_Environment:
    def __init__(self):
        # Initialize global variables-------------------------------------------------
        self.data_ref = 0
        self.saved_dat_ref = 0
        self.format_data = DataManagement(cons.trainFile, cons.testFile)

    def getTrainInstance(self):
        """ Returns the current training instance. """
        self.train_inst_condition = self.format_data.trainFormatted[self.data_ref][0]
        self.train_inst_phenotype = self.format_data.trainFormatted[self.data_ref][1]
        if self.data_ref < (self.format_data.numTrainphenotypes - 1):
            self.data_ref += 1
        else:  # Once learning has completed an epoch (i.e. a cycle of iterations though the entire training dataset) it starts back at the first instance in the data)
            self.data_ref = 0
        return [self.train_inst_condition, self.train_inst_phenotype]

    def getTestInstance(self):
        """ Returns the current training instance. """
        self.test_inst_condition = self.format_data.formatted_test_data[self.data_ref][0]
        self.test_inst_phenotype = self.format_data.formatted_test_data[self.data_ref][1]
        if self.data_ref < (self.format_data.numTestphenotypes - 1):
            self.data_ref += 1
        else:
            self.data_ref = 0
        return [self.test_inst_condition, self.test_inst_phenotype]

    def resetDataRef(self, is_train):
        """ Resets the environment back to the first instance in the current data set. """
        self.data_ref = 0

    def startEvaluationMode(self):
        """ Turns on evaluation mode.  Saves the instance we left off in the training data. """
        self.saved_dat_ref = self.data_ref

    def stopEvaluationMode(self):
        """ Turns off evaluation mode.  Re-establishes place in dataset."""
        self.data_ref = self.saved_dat_ref
