"""
Name:        UCS_Offline_Environment.py
Authors:     Ryan Urbanowicz - Written at Dartmouth College, Hanover, NH, USA
Contact:     ryan.j.urbanowicz@darmouth.edu
Created:     November 1, 2013
Description: In the context of data mining and classification tasks, the environment is a data set with a limited number of phenotypes with X attributes
             and some endpoint (typically a discrete phenotype or class) of interest.  This module loads the data set, automatically detects features of the data.
             
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

from UCS.UCS_Constants import *
# Import Required Modules--------------------------------------
from UCS.UCS_DataManagement import DataManagement


# -------------------------------------------------------------

class Offline_Environment:
    def __init__(self):
        # Initialize global variables-------------------------------------------------
        self.dataRef = 0
        self.storeDataRef = 0
        self.formatData = DataManagement(cons.trainFile, cons.testFile)

        # Initialize the first dataset phenotype to be passed to LCS
        self.currentTrainState = self.formatData.trainFormatted[self.dataRef][0]
        self.currentTrainphenotype = self.formatData.trainFormatted[self.dataRef][1]
        if cons.testFile == 'None':
            pass
        else:
            self.currentTestState = self.formatData.testFormatted[self.dataRef][0]
            self.currentTestphenotype = self.formatData.testFormatted[self.dataRef][1]

    def getTrainphenotype(self):
        """ Returns the current training phenotype. """
        return [self.currentTrainState, self.currentTrainphenotype]

    def getTestphenotype(self):
        """ Returns the current training phenotype. """
        return [self.currentTestState, self.currentTestphenotype]

    def newphenotype(self, isTraining):
        """  Shifts the environment to the next phenotype in the data. """
        # -------------------------------------------------------
        # Training Data
        # -------------------------------------------------------
        if isTraining:
            if self.dataRef < (self.formatData.numTrainphenotypes - 1):
                self.dataRef += 1
                self.currentTrainState = self.formatData.trainFormatted[self.dataRef][0]
                self.currentTrainphenotype = self.formatData.trainFormatted[self.dataRef][1]
            else:  # Once learning has completed an epoch (i.e. a cycle of iterations though the entire training dataset) it starts back at the first phenotype in the data)
                self.resetDataRef(isTraining)

        # -------------------------------------------------------
        # Testing Data
        # -------------------------------------------------------
        else:
            if self.dataRef < (self.formatData.numTestphenotypes - 1):
                self.dataRef += 1
                self.currentTestState = self.formatData.testFormatted[self.dataRef][0]
                self.currentTestphenotype = self.formatData.testFormatted[self.dataRef][1]

    def resetDataRef(self, isTraining):
        """ Resets the environment back to the first phenotype in the current data set. """
        self.dataRef = 0
        if isTraining:
            self.currentTrainState = self.formatData.trainFormatted[self.dataRef][0]
            self.currentTrainphenotype = self.formatData.trainFormatted[self.dataRef][1]
        else:
            self.currentTestState = self.formatData.testFormatted[self.dataRef][0]
            self.currentTestphenotype = self.formatData.testFormatted[self.dataRef][1]

    def startEvaluationMode(self):
        """ Turns on evaluation mode.  Saves the phenotype we left off in the training data. """
        self.storeDataRef = self.dataRef

    def stopEvaluationMode(self):
        """ Turns off evaluation mode.  Re-establishes place in dataset."""
        self.dataRef = self.storeDataRef
