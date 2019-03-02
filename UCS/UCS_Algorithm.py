"""
Name:        UCS_Algorithm.py
Authors:     Ryan Urbanowicz - Written at Dartmouth College, Hanover, NH, USA
Contact:     ryan.j.urbanowicz@darmouth.edu
Created:     November 1, 2013
Description: The major controlling module of LCS.  Includes the major run loop which controls learning over a specified number of iterations.  Also includes
             periodic tracking of estimated performance, and checkpoints where complete evaluations of the LCS rule population is performed.
             
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

import math
import random
import time

from UCS.UCS_ClassAccuracy import ClassAccuracy
# Import Required Modules-------------------------------
from UCS.UCS_ClassifierSet import ClassifierSet
from UCS.UCS_OutputFileManager import OutputFileManager
from UCS.UCS_Prediction import *


# ------------------------------------------------------

class UCS:
    standardAccuracy = 0

    def __init__(self):
        """ Initializes the LCS algorithm """
        print("UCS: Initializing Algorithm...")
        # Global Parameters-------------------------------------------------------------------------------------
        self.population = None  # The rule population (the 'solution/model' evolved by LCS)
        self.learnTrackOut = None  # Output file that will store tracking information during learning

        # -------------------------------------------------------
        # POPULATION REBOOT - Begin LCS learning from an existing saved rule population
        # -------------------------------------------------------
        if cons.doPopulationReboot:
            self.populationReboot()

        # -------------------------------------------------------
        # NORMAL LCS - Run LCS from scratch on given data
        # -------------------------------------------------------
        else:
            try:
                self.learnTrackOut = open(cons.outFileName + '_LearnTrack.txt', 'w')
            except Exception as inst:
                print(type(inst))
                print(inst.args)
                print(inst)
                print('cannot open', cons.outFileName + '_LearnTrack.txt')
                raise
            else:
                self.learnTrackOut.write(
                    "Explore_Iteration\tMacroPopSize\tMicroPopSize\tAccuracy_Estimate\tAveGenerality\tExpRules\tTime(min)\n")

            # Instantiate Population---------
            self.population = ClassifierSet()
            self.exploreIter = 0
            self.correct = [0.0 for i in range(cons.trackingFrequency)]

        # Run the LCS algorithm-------------------------------------------------------------------------------
        self.run_UCS()

    def run_UCS(self):

        trackedAccuracy = 0

        """ Runs the initialized LCS algorithm. """
        # --------------------------------------------------------------
        print("Learning Checkpoints: " + str(cons.learningCheckpoints))
        print("Maximum Iterations: " + str(cons.maxLearningIterations))
        print("Beginning LCS learning iterations.")
        print(
            "------------------------------------------------------------------------------------------------------------------------------------------------------")

        # -------------------------------------------------------
        # MAJOR LEARNING LOOP
        # -------------------------------------------------------
        t0 = time.clock()
        while self.exploreIter < cons.maxLearningIterations: #and trackedAccuracy < 1:

            # -------------------------------------------------------
            # GET NEW phenotype AND RUN A LEARNING ITERATION
            # -------------------------------------------------------
            phenotype = cons.env.getTrainphenotype()
            self.runIteration(phenotype, self.exploreIter)

            # -------------------------------------------------------------------------------------------------------------------------------
            # EVALUATIONS OF ALGORITHM
            # -------------------------------------------------------------------------------------------------------------------------------
            cons.timer.startTimeEvaluation()

            # -------------------------------------------------------
            # TRACK LEARNING ESTIMATES
            # -------------------------------------------------------
            if (self.exploreIter % cons.trackingFrequency) == (cons.trackingFrequency - 1) and self.exploreIter > 0:

                self.population.runPopAveEval(self.exploreIter)
                trackedAccuracy = sum(self.correct) / float(
                    cons.trackingFrequency)  # Accuracy over the last "trackingFrequency" number of iterations.
                self.learnTrackOut.write(self.population.getPopTrack(trackedAccuracy, self.exploreIter + 1, cons.trackingFrequency))  # Report learning progress to standard out and tracking file.

            cons.timer.stopTimeEvaluation()

            # -------------------------------------------------------
            # CHECKPOINT - COMPLETE EVALUTATION OF POPULATION - strategy different for discrete vs continuous phenotypes
            # -------------------------------------------------------
            if (self.exploreIter + 1) in cons.learningCheckpoints:
                self.evaluate()

            # -------------------------------------------------------
            # ADJUST MAJOR VALUES FOR NEXT ITERATION
            # -------------------------------------------------------
            self.exploreIter += 1  # Increment current learning iteration
            cons.env.newphenotype(True)  # Step to next phenotype in training set

        # Once LCS has reached the last learning iteration, close the tracking file
        self.learnTrackOut.close()
        self.evaluate()

        t1 = time.clock()
        total = t1-t0
        print("Run time in seconds: %.2f" % round(total,2))
        print("LCS Run Complete")

    def evaluate(self):
        cons.timer.startTimeEvaluation()
        print(
            "------------------------------------------------------------------------------------------------------------------------------------------------------")
        print("Running Population Evaluation after " + str(self.exploreIter + 1) + " iterations.")
        self.population.runPopAveEval(self.exploreIter)
        self.population.runAttGeneralitySum(True)
        cons.env.startEvaluationMode()  # Preserves learning position in training data
        if cons.testFile != 'None' or cons.kfold > 0:  # If a testing file is available.
            if cons.env.formatData.discretephenotype:
                trainEval = self.doPopEvaluation(True)
                testEval = self.doPopEvaluation(False)
        else:  # Only a training file is available
            if cons.env.formatData.discretephenotype:
                trainEval = self.doPopEvaluation(True)
                testEval = None
            else:
                testEval = None
        cons.env.stopEvaluationMode()  # Returns to learning position in training data
        cons.timer.stopTimeEvaluation()
        cons.timer.returnGlobalTimer()

        # Write output files----------------------------------------------------------------------------------------------------------
        OutputFileManager().writePopStats(cons.outFileName, trainEval, testEval, self.exploreIter + 1,
                                          self.population, self.correct)
        OutputFileManager().writePop(cons.outFileName, self.exploreIter + 1, self.population)
        # ----------------------------------------------------------------------------------------------------------------------------
        print("Continue Learning...")
        print(
            "--------------------------------------------------------------------------------------------------")

    def runIteration(self, phenotype, exploreIter):
        """ Run a single LCS learning iteration. """
        # -----------------------------------------------------------------------------------------------------------------------------------------
        # FORM A MATCH SET - includes covering
        # -----------------------------------------------------------------------------------------------------------------------------------------
        self.population.makeMatchSet(phenotype, exploreIter)
        # -----------------------------------------------------------------------------------------------------------------------------------------
        # MAKE A PREDICTION - utilized here for tracking estimated learning progress.  Typically used in the explore phase of many LCS algorithms.
        # -----------------------------------------------------------------------------------------------------------------------------------------
        cons.timer.startTimeEvaluation()
        prediction = Prediction(self.population)
        phenotypePrediction = prediction.getDecision()

        # -------------------------------------------------------
        # PREDICTION NOT POSSIBLE
        # -------------------------------------------------------
        if phenotypePrediction is None or phenotypePrediction == 'Tie':
            if cons.env.formatData.discretephenotype:
                self.phenotypePrediction = random.choice(cons.env.formatData.phenotypeList)
            else:
                self.phenotypePrediction = random.randrange(cons.env.formatData.phenotypeList[0],
                                                           cons.env.formatData.phenotypeList[1], (
                                                                   cons.env.formatData.phenotypeList[1] -
                                                                   cons.env.formatData.phenotypeList[0]) / float(1000))
        else:  # Prediction Successful
            # -------------------------------------------------------
            # DISCRETE phenotype PREDICTION
            # -------------------------------------------------------
            if cons.env.formatData.discretephenotype:
                if phenotypePrediction == phenotype[1]:
                    self.correct[exploreIter % cons.trackingFrequency] = 1
                else:
                    self.correct[exploreIter % cons.trackingFrequency] = 0
            # -------------------------------------------------------
            # CONTINUOUS phenotype PREDICTION
            # -------------------------------------------------------
            else:
                predictionError = math.fabs(phenotypePrediction - float(phenotype[1]))
                phenotypeRange = cons.env.formatData.phenotypeList[1] - cons.env.formatData.phenotypeList[0]
                accuracyEstimate = 1.0 - (predictionError / float(phenotypeRange))
                self.correct[exploreIter % cons.trackingFrequency] = accuracyEstimate
        cons.timer.stopTimeEvaluation()
        # -----------------------------------------------------------------------------------------------------------------------------------------
        # FORM A CORRECT SET
        # -----------------------------------------------------------------------------------------------------------------------------------------
        self.population.makeCorrectSet(phenotype[1])
        # -----------------------------------------------------------------------------------------------------------------------------------------
        # UPDATE PARAMETERS
        # -----------------------------------------------------------------------------------------------------------------------------------------
        self.population.updateParams(exploreIter)
        # -----------------------------------------------------------------------------------------------------------------------------------------
        # SUBSUMPTION - APPLIED TO CORRECT SET - A heuristic for addition additional generalization pressure to LCS
        # -----------------------------------------------------------------------------------------------------------------------------------------
        if cons.doSubsumption:
            cons.timer.startTimeSubsumption()
            self.population.doCorrectSetSubsumption()
            cons.timer.stopTimeSubsumption()
        # -----------------------------------------------------------------------------------------------------------------------------------------
        # RUN THE GENETIC ALGORITHM - Discover new offspring rules from a selected pair of parents
        # -----------------------------------------------------------------------------------------------------------------------------------------
        self.population.runGA(exploreIter, phenotype[0], phenotype[1])
        # -----------------------------------------------------------------------------------------------------------------------------------------
        # SELECT RULES FOR DELETION - This is done whenever there are more rules in the population than 'N', the maximum population size.
        # -----------------------------------------------------------------------------------------------------------------------------------------
        self.population.deletion(exploreIter)
        self.population.clearSets()  # Clears the match and correct sets for the next learning iteration

    def doPopEvaluation(self, isTrain):
        """ Performs a complete evaluation of the current rule population.  The population is unchanged throughout this evaluation. Works on both training and testing data. """
        if isTrain:
            myType = "TRAINING"
        else:
            myType = "TESTING"
        noMatch = 0  # How often does the population fail to have a classifier that matches an phenotype in the data.
        tie = 0  # How often can the algorithm not make a decision between classes due to a tie.
        cons.env.resetDataRef(isTrain)  # Go to the first phenotype in dataset
        phenotypeList = cons.env.formatData.phenotypeList
        # ----------------------------------------------
        classAccDict = {}
        for each in phenotypeList:
            classAccDict[each] = ClassAccuracy()
        # ----------------------------------------------
        if isTrain:
            phenotypes = cons.env.formatData.numTrainphenotypes
        else:
            phenotypes = cons.env.formatData.numTestphenotypes
        # ----------------------------------------------------------------------------------------------
        for inst in range(phenotypes):
            if isTrain:
                phenotype = cons.env.getTrainphenotype()
            else:
                phenotype = cons.env.getTestphenotype()
            # -----------------------------------------------------------------------------
            self.population.makeEvalMatchSet(phenotype[0])
            prediction = Prediction(self.population)
            phenotypeSelection = prediction.getDecision()
            # -----------------------------------------------------------------------------

            if phenotypeSelection is None:
                noMatch += 1
            elif phenotypeSelection == 'Tie':
                tie += 1
            else:  # phenotypes which failed to be covered are excluded from the accuracy calculation
                for each in phenotypeList:
                    thisIsMe = False
                    accuratephenotype = False
                    truephenotype = phenotype[1]
                    if each == truephenotype:
                        thisIsMe = True
                    if phenotypeSelection == truephenotype:
                        accuratephenotype = True
                    classAccDict[each].updateAccuracy(thisIsMe, accuratephenotype)

            cons.env.newphenotype(isTrain)  # next phenotype
            self.population.clearSets()
            # ----------------------------------------------------------------------------------------------
        # Calculate Standard Accuracy--------------------------------------------
        phenotypesCorrectlyClassified = classAccDict[phenotypeList[0]].T_myClass + classAccDict[
            phenotypeList[0]].T_otherClass
        phenotypesIncorrectlyClassified = classAccDict[phenotypeList[0]].F_myClass + classAccDict[
            phenotypeList[0]].F_otherClass
        self.standardAccuracy = float(phenotypesCorrectlyClassified) / float(
            phenotypesCorrectlyClassified + phenotypesIncorrectlyClassified)

        # Calculate Balanced Accuracy---------------------------------------------
        T_mySum = 0
        T_otherSum = 0
        F_mySum = 0
        F_otherSum = 0
        for each in phenotypeList:
            T_mySum += classAccDict[each].T_myClass
            T_otherSum += classAccDict[each].T_otherClass
            F_mySum += classAccDict[each].F_myClass
            F_otherSum += classAccDict[each].F_otherClass
        balancedAccuracy = ((0.5 * T_mySum / (float(T_mySum + F_otherSum)) + 0.5 * T_otherSum / (
            float(T_otherSum + F_mySum))))  # BalancedAccuracy = (Specificity + Sensitivity)/2

        # Adjustment for uncovered phenotypes - to avoid positive or negative bias we incorporate the probability of guessing a phenotype by chance (e.g. 50% if two phenotypes)
        predictionFail = float(noMatch) / float(phenotypes)
        predictionTies = float(tie) / float(phenotypes)
        phenotypeCoverage = 1.0 - predictionFail
        predictionMade = 1.0 - (predictionFail + predictionTies)

        adjustedStandardAccuracy = (self.standardAccuracy * predictionMade) + (
                (1.0 - predictionMade) * (1.0 / float(len(phenotypeList))))
        adjustedBalancedAccuracy = (balancedAccuracy * predictionMade) + (
                (1.0 - predictionMade) * (1.0 / float(len(phenotypeList))))

        # Adjusted Balanced Accuracy is calculated such that phenotypes that did not match have a consistent probability of being correctly classified in the reported accuracy.
        print("-----------------------------------------------")
        print(str(myType) + " Accuracy Results:-------------")
        print("Instance Coverage = " + str(phenotypeCoverage * 100.0) + '%')
        print("Prediction Ties = " + str(predictionTies * 100.0) + '%')
        print(str(phenotypesCorrectlyClassified) + ' out of ' + str(
            phenotypes) + ' instances covered and correctly classified.')
        print("Standard Accuracy = " + str(self.standardAccuracy))
        print("Standard Accuracy (Adjusted) = " + str(adjustedStandardAccuracy))
        print("Balanced Accuracy (Adjusted) = " + str(adjustedBalancedAccuracy))
        UCS.standardAccuracy = self.standardAccuracy
        # Balanced and Standard Accuracies will only be the same when there are equal phenotypes representative of each phenotype AND there is 100% covering.
        resultList = [adjustedBalancedAccuracy, phenotypeCoverage]
        return resultList

    def populationReboot(self):
        """ Manages the reformation of a previously saved LCS classifier population. """
        cons.timer.setTimerRestart(cons.popRebootPath)  # Rebuild timer objects
        # --------------------------------------------------------------------
        try:  # Re-open track learning file for continued tracking of progress.
            self.learnTrackOut = open(cons.outFileName + '_LearnTrack.txt', 'a')
        except Exception as inst:
            print(type(inst))
            print(inst.args)
            print(inst)
            print('cannot open', cons.outFileName + '_LearnTrack.txt')
            raise

            # Extract last iteration from file name---------------------------------------------
        temp = cons.popRebootPath.split('_')
        iterRef = len(temp) - 1
        completedIterations = int(temp[iterRef])
        print("Rebooting rule population after " + str(completedIterations) + " iterations.")
        self.exploreIter = completedIterations - 1
        for i in range(len(cons.learningCheckpoints)):
            cons.learningCheckpoints[i] += completedIterations
        cons.maxLearningIterations += completedIterations

        # Rebuild existing population from text file.--------
        self.population = ClassifierSet(cons.popRebootPath)
        # ---------------------------------------------------
        try:  # Obtain correct track
            f = open(cons.popRebootPath + "_PopStats.txt", 'r')
        except Exception as inst:
            print(type(inst))
            print(inst.args)
            print(inst)
            print('cannot open', cons.popRebootPath + "_PopStats.txt")
            raise
        else:
            correctRef = 26  # File reference position
            tempLine = None
            for i in range(correctRef):
                tempLine = f.readline()
            tempList = tempLine.strip().split('\t')
            self.correct = tempList
            if cons.env.formatData.discretephenotype:
                for i in range(len(self.correct)):
                    self.correct[i] = int(self.correct[i])
            else:
                for i in range(len(self.correct)):
                    self.correct[i] = float(self.correct[i])
            f.close()
