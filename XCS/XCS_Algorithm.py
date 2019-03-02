# Import Required Modules-------------------------------

from XCS.XCS_ClassAccuracy import ClassAccuracy
from XCS.XCS_ClassifierSet import ClassifierSet
from XCS.XCS_OutputFileManager import OutputFileManager
from XCS.XCS_Prediction import *


# ------------------------------------------------------
class XCS:
    standard_accuracy = 0

    def __init__(self):
        """ Initializes the XCS algorithm """
        print("XCS: Initializing Algorithm...")
        # Global Parameters-------------------------------------------------------------------------------------
        self.population = None  # The rule population (the 'solution/model' evolved by XCS)
        self.learnTrackOut = None  # Output file that will store tracking information during learning

        # -------------------------------------------------------
        # POPULATION REBOOT - Begin XCS learning from an existing saved rule population
        # -------------------------------------------------------
        if cons.doPopulationReboot:
            self.populationReboot()
            print("population rebooted")

        # -------------------------------------------------------
        # NORMAL XCS - Run XCS from scratch on given data
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
            self.iteration = 0
            self.trackedResults = []

    def run_XCS(self):
        """ Runs the initialized XCS algorithm. """
        # --------------------------------------------------------------
        trackedAccuracy = 0
        print("Learning Checkpoints: " + str(cons.learningCheckpoints))
        print("Maximum Iterations: " + str(cons.maxLearningIterations))
        print("Beginning XCS learning iterations.")
        print(
            "------------------------------------------------------------------------------------------------------------------------------------------------------")
        exploreMode = 1
        # -------------------------------------------------------
        # MAJOR LEARNING LOOP
        # -------------------------------------------------------
        while self.iteration < cons.maxLearningIterations: #and trackedAccuracy < 1: #possibly comment this out?! Continue learning even if accuracy = 1
            # -------------------------------------------------------
            # GET NEW INSTANCE AND RUN A LEARNING ITERATION
            # -------------------------------------------------------
            self.selectAction(exploreMode)
            self.iteration += 1  # Increment current learning iteration

            # -------------------------------------------------------------------------------------------------------------------------------
            # EVALUATIONS OF ALGORITHM
            # -------------------------------------------------------------------------------------------------------------------------------
            cons.timer.startTimeEvaluation()

            # -------------------------------------------------------
            # TRACK LEARNING ESTIMATES
            # -------------------------------------------------------
            if self.iteration % cons.trackingFrequency == 0:
                self.population.runPopAveEval()
                trackedAccuracy = sum(self.trackedResults) / float(
                    len(self.trackedResults))  # Accuracy over the last "trackingFrequency" number of iterations.
                self.trackedResults = []
                self.learnTrackOut.write(self.population.getPopTrack(trackedAccuracy, self.iteration,
                                                                     cons.trackingFrequency))  # Report learning progress to standard out and tracking file.
            cons.timer.stopTimeEvaluation()

            # -------------------------------------------------------
            # CHECKPOINT - COMPLETE EVALUATION OF POPULATION - strategy different for discrete vs continuous phenotypes
            # -------------------------------------------------------
            if self.iteration in cons.learningCheckpoints:
                cons.timer.startTimeEvaluation()
                print(
                    "------------------------------------------------------------------------------------------------------------------------------------------------------")
                print("Running Population Evaluation after " + str(self.iteration) + " iterations.")

                self.population.runPopAveEval()
                self.population.runAttGeneralitySum(True)
                cons.env.startEvaluationMode()  # Preserves learning position in training data
                if cons.testFile != 'None' or cons.kfold > 0:  # If a testing file is available.
                    if cons.env.format_data.discretephenotype:
                        train_eval = self.doPopEvaluation(True)
                        test_eval = self.doPopEvaluation(False)
                else:  # Only a training file is available
                    if cons.env.format_data.discretephenotype:
                        train_eval = self.doPopEvaluation(True)
                        test_eval = None
                    else:
                        test_eval = None

                cons.env.stopEvaluationMode()  # Returns to learning position in training data
                cons.timer.stopTimeEvaluation()
                cons.timer.returnGlobalTimer()

                # Write output files----------------------------------------------------------------------------------------------------------
                OutputFileManager().writePopStats(cons.outFileName, train_eval, test_eval, self.iteration,
                                                  self.population, self.trackedResults)
                OutputFileManager().writePop(cons.outFileName, self.iteration, self.population)
                # ----------------------------------------------------------------------------------------------------------------------------

                print("Continue Learning...")
                print(
                    "------------------------------------------------------------------------------------------------------------------------------------------------------")

            # Switch between explore and exploit
            if cons.exploration == 0.5:
                exploreMode = 1 - exploreMode

        # Once XCS has reached the last learning iteration, close the tracking file
        self.learnTrackOut.close()
        print("XCS Run Complete")
        self.population.finalise()
        ret_eval = []
        if cons.testFile != 'None' or cons.kfold > 0:  # If a testing file is available.
            if cons.env.format_data.discretephenotype:
                train_eval = self.doPopEvaluation(True)
                test_eval = self.doPopEvaluation(False)
            ret_eval += test_eval
        else:  # Only a training file is available
            if cons.env.format_data.discretephenotype:
                ret_eval = train_eval = self.doPopEvaluation(True)
                test_eval = None
            ret_eval += train_eval
        OutputFileManager().writePopStats(cons.outFileName + '_finalised', train_eval, test_eval, self.iteration,
                                          self.population, self.trackedResults)
        OutputFileManager().writePop(cons.outFileName + '_finalised', self.iteration, self.population)
        return ret_eval

    def selectAction(self, exploreMode):
        phenotype = cons.env.getTrainInstance()
        if exploreMode == 1:
            self.runExplore(phenotype)
        else:
            self.runExploit(phenotype)

    def runExploit(self, phenotype):
        """ Run an exploit iteration. """
        self.population.generateMatchSet(phenotype[0], self.iteration)
        cons.timer.startTimeEvaluation()
        prediction = Prediction(self.population)
        selectedAction = prediction.decide(exploring=False)
        if selectedAction == phenotype[1]:
            reward = 1000.0
            self.trackedResults.append(1)
        else:
            reward = 0.0
            self.trackedResults.append(0)
        cons.timer.stopTimeEvaluation()
        self.population.generateActionSet(selectedAction)
        self.population.updateSets(reward)
        self.population.clearSets()  # Clears the match and action sets for the next learning iteration

    def runExplore(self, phenotype):
        """ Run an explore learning iteration. """
        reward = 0.0
        # -----------------------------------------------------------------------------------------------------------------------------------------
        # FORM A MATCH SET - includes covering
        # -----------------------------------------------------------------------------------------------------------------------------------------
        self.population.generateMatchSet(phenotype[0], self.iteration)
        # -----------------------------------------------------------------------------------------------------------------------------------------
        # MAKE A PREDICTION - utilized here for tracking estimated learning progress.  Typically used in the explore phase of many LCS algorithms.
        # -----------------------------------------------------------------------------------------------------------------------------------------
        cons.timer.startTimeEvaluation()
        prediction = Prediction(self.population)
        selectedAction = prediction.decide(exploring=True)
        # -------------------------------------------------------
        # DISCRETE PHENOTYPE PREDICTION
        # -------------------------------------------------------
        if selectedAction == phenotype[1]:
            reward = 1000.0

        cons.timer.stopTimeEvaluation()
        # -----------------------------------------------------------------------------------------------------------------------------------------
        # FORM AN ACTION SET
        # -----------------------------------------------------------------------------------------------------------------------------------------
        self.population.generateActionSet(selectedAction)
        # -----------------------------------------------------------------------------------------------------------------------------------------
        # UPDATE PARAMETERS
        # -----------------------------------------------------------------------------------------------------------------------------------------
        self.population.updateSets(reward)
        # -----------------------------------------------------------------------------------------------------------------------------------------
        # SUBSUMPTION - APPLIED TO MATCH SET - A heuristic for addition additional generalization pressure to XCS
        # -----------------------------------------------------------------------------------------------------------------------------------------
        if cons.doActionSetSubsumption:
            cons.timer.startTimeSubsumption()
            self.population.doActionSetSubsumption()
            cons.timer.stopTimeSubsumption()
        # -----------------------------------------------------------------------------------------------------------------------------------------
        # RUN THE GENETIC ALGORITHM - Discover new offspring rules from a selected pair of parents
        # -----------------------------------------------------------------------------------------------------------------------------------------
        self.population.runGA(self.iteration, phenotype[0])
        self.population.clearSets() #Clears the match and action sets (done in runGA)

    def doPopEvaluation(self, is_train):
        """ Performs a complete evaluation of the current rule population.  The population is unchanged throughout this evaluation. Works on both training and testing data. """
        if is_train:
            instances = cons.env.format_data.numTrainphenotypes
            my_type = "TRAINING"
        else:
            instances = cons.env.format_data.numTestphenotypes
            my_type = "TESTING"
        no_match = 0  # How often does the population fail to have a classifier that matches an instance in the data.
        tie = 0  # How often can the algorithm not make a decision between classes due to a tie.
        cons.env.resetDataRef(is_train)  # Go to the first instance in dataset
        phenotype_list = cons.env.format_data.phenotypeList
        # ----------------------------------------------
        class_accuracies = {}
        for each in phenotype_list:
            class_accuracies[each] = ClassAccuracy()
        # ----------------------------------------------------------------------------------------------
        for _ in range(instances):
            if is_train:
                phenotype = cons.env.getTrainInstance()
            else:
                phenotype = cons.env.getTestInstance()
            # -----------------------------------------------------------------------------
            self.population.makeEvalMatchSet(phenotype[0])
            prediction = Prediction(self.population)
            selected_action = prediction.decide(exploring=False)
            # -----------------------------------------------------------------------------

            if selected_action is None:
                no_match += 1
            elif selected_action == 'Tie':
                tie += 1
            else:  # Instances which failed to be covered are excluded from the accuracy calculation
                for each in phenotype_list:
                    is_correct = False
                    accurate_action = False
                    right_action = phenotype[1]
                    if each == right_action:
                        is_correct = True
                    if selected_action == right_action:
                        accurate_action = True
                    class_accuracies[each].updateAccuracy(is_correct, accurate_action)

            self.population.clearSets()
        # ----------------------------------------------------------------------------------------------
        # Calculate Standard Accuracy--------------------------------------------
        correct_cases = class_accuracies[phenotype_list[0]].T_myClass + class_accuracies[phenotype_list[0]].T_otherClass
        incorrect_cases = class_accuracies[phenotype_list[0]].F_myClass + class_accuracies[
            phenotype_list[0]].F_otherClass
        accuracy = float(correct_cases) / float(correct_cases + incorrect_cases)

        # Calculate Balanced Accuracy---------------------------------------------
        T_mySum = 0
        T_otherSum = 0
        F_mySum = 0
        F_otherSum = 0
        for each in phenotype_list:
            T_mySum += class_accuracies[each].T_myClass
            T_otherSum += class_accuracies[each].T_otherClass
            F_mySum += class_accuracies[each].F_myClass
            F_otherSum += class_accuracies[each].F_otherClass
        balanced_accuracy = ((0.5 * T_mySum / (float(T_mySum + F_otherSum)) + 0.5 * T_otherSum / (
            float(T_otherSum + F_mySum))))  # BalancedAccuracy = (Specificity + Sensitivity)/2

        # Adjustment for uncovered instances - to avoid positive or negative bias we incorporate the probability of guessing a phenotype by chance (e.g. 50% if two phenotypes)
        prediction_fail = float(no_match) / float(instances)
        prediction_ties = float(tie) / float(instances)
        covered_instances = 1.0 - prediction_fail
        prediction_made = 1.0 - (prediction_fail + prediction_ties)

        self.standard_accuracy = accuracy * prediction_made
        adjusted_accuracy = self.standard_accuracy + ((1.0 - prediction_made) * (1.0 / float(len(phenotype_list))))
        adjusted_balanced_accuracy = (balanced_accuracy * prediction_made) + (
                    (1.0 - prediction_made) * (1.0 / float(len(phenotype_list))))

        # Adjusted Balanced Accuracy is calculated such that instances that did not match have a consistent probability of being correctly classified in the reported accuracy.
        print("-----------------------------------------------")
        print(str(my_type) + " Accuracy Results:-------------")
        print("Instance Coverage = " + str(covered_instances * 100.0) + '%')
        print("Prediction Ties = " + str(prediction_ties * 100.0) + '%')
        print(str(correct_cases) + ' out of ' + str(instances) + ' instances covered and correctly classified.')
        print("Standard Accuracy = " + str(self.standard_accuracy))
        print("Standard Accuracy (Adjusted) = " + str(adjusted_accuracy))
        print("Balanced Accuracy (Adjusted) = " + str(adjusted_balanced_accuracy))
        XCS.standard_accuracy = self.standard_accuracy
        # Balanced and Standard Accuracies will only be the same when there are equal instances representative of each phenotype AND there is 100% covering.

        return [adjusted_accuracy, self.standard_accuracy, adjusted_balanced_accuracy, covered_instances]

    def populationReboot(self):
        """ Manages the reformation of a previously saved XCS classifier population. """
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
        iter_ref = len(temp) - 1
        completed_iterations = int(temp[iter_ref])
        print("Rebooting rule population after " + str(completed_iterations) + " iterations.")
        self.iteration = completed_iterations - 1
        for i in range(len(cons.learningCheckpoints)):
            cons.learningCheckpoints[i] += completed_iterations
        cons.maxLearningIterations += completed_iterations

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
            correct_ref = 26  # File reference position
            temp_line = None
            for i in range(correct_ref):
                temp_line = f.readline()
            temp_list = temp_line.strip().split('\t')
            self.trackedResults = temp_list
            if cons.env.format_data.discretephenotype:
                for i in range(len(self.trackedResults)):
                    self.trackedResults[i] = int(self.trackedResults[i])
            else:
                for i in range(len(self.trackedResults)):
                    self.trackedResults[i] = float(self.trackedResults[i])
            f.close()
