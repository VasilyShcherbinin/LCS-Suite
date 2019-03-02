"""
Name:        UCS_Run.py
Authors:     Ryan Urbanowicz - Written at Dartmouth College, Hanover, NH, USA
Contact:     ryan.j.urbanowicz@darmouth.edu
Created:     November 1, 2013
Description: To run e-LCS, run this module.  A properly formatted configuration file, including all run parameters must be included with the path to that 
             file given below.  In this example, the configuration file has been included locally, so only the file name is required.
             
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
# Import Required Modules------------------------------------
import time

from UCS.UCS_Timer import Timer
from UCS.UCS_Algorithm import UCS
from UCS.UCS_OutputFileManager import OutputFileManager
from UCS.UCS_ConfigParser import ConfigParser
from UCS.UCS_Constants import *
from UCS.UCS_Offline_Environment import Offline_Environment
# -----------------------------------------------------------

def mainRun():

    # Specify the name and file path for the configuration file.
    configurationFile = "UCS_Configuration_File.txt"
    # Obtain all run parameters from the configuration file and store them in the 'Constants' module.
    ConfigParser(configurationFile)
    # Initialize the 'Timer' module which tracks the run time of algorithm and it's different components.
    timer = Timer()
    cons.referenceTimer(timer)
    # Initialize the 'Environment' module which manages the data presented to the algorithm.  While e-LCS learns iteratively (one inistance at a time
    env = Offline_Environment()
    cons.referenceEnv(
        env)  # Passes the environment to 'Constants' (cons) so that it can be easily accessed from anywhere within the code.
    cons.parseIterations()  # Identify the maximum number of learning iterations as well as evaluation checkpoints.
    # Clear Local_Output Folder before Run
    folder = 'Local_Output'
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)
    # Run the e-LCS algorithm.
    # UCS()
    kfold_accuracy = 0
    t0 = time.clock()
    if cons.kfold > 0:
        total_instances = env.formatData.numTrainphenotypes
        env.formatData.splitDataIntoKSets()
        accurate_numbs = [0.0] * cons.kfold
        for i in range(cons.kfold):
            print("")
            print("Starting next UCS learning iteration...")
            print("K-FOLD: " + str(i))
            env.formatData.selectTrainTestSets(i)
            cons.parseIterations()  # Identify the maximum number of learning iterations as well as evaluation checkpoints.
            UCS()
            accuracy = UCS.standardAccuracy
            accurate_numbs[i] = accuracy * env.formatData.numTestphenotypes
            kfold_accuracy = sum(accurate_numbs) / total_instances
            print("AVERAGE ACCURACY AFTER " + str(cons.kfold) + "-FOLD CROSS VALIDATION is " + str(
                kfold_accuracy))
    else:
        cons.parseIterations()  # Identify the maximum number of learning iterations as well as evaluation checkpoints.
        UCS()
    t1 = time.clock()
    total = t1 - t0
    total = round(total, 2)
    print("Total run time in seconds: %.2f" % total)
    f = open("RESULTS_FILE.txt", 'a')
    f.write(" Accuracy: " + str(kfold_accuracy) + " Total time: " + str(total) + " Rules: " + str(OutputFileManager.totalPopulationSize) + "\n")

if __name__ == '__main__':

    for i in range(1):
        mainRun()



