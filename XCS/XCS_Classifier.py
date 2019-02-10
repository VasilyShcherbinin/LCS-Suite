"""
Name:        XCS_Classifier.py
Authors:     Bao Trung
Contact:     baotrung@ecs.vuw.ac.nz
Created:     July, 2017
Description:

---------------------------------------------------------------------------------------------------------------------------------------------------------
XCS: Michigan-style Learning Classifier System - A LCS for Reinforcement Learning.  This XCS follows the version descibed in "An Algorithmic Description of XCS" published by Martin Butz and Stewart Wilson (2002).

---------------------------------------------------------------------------------------------------------------------------------------------------------
"""

import random

# Import Required Modules---------------
from XCS.XCS_Constants import *


# --------------------------------------

class Classifier:
    def __init__(self, a=None, b=None, c=None):
        # Major Parameters --------------------------------------------------
        self.specifiedAttributes = []  # Attribute Specified in classifier: Similar to Bacardit 2009 - ALKR + GABIL, continuous and discrete rule representation
        self.condition = []  # States of Attributes Specified in classifier: Similar to Bacardit 2009 - ALKR + GABIL, continuous and discrete rule representation
        self.action = None  # Class if the endpoint is discrete, and a continuous phenotype if the endpoint is continuous

        self.prediction = cons.init_pred  # Classifier payoff - initialized to a constant initial payoff value
        self.error = cons.init_err  # Classifier error - initialized to a constant initial error value
        self.fitness = cons.init_fit  # Classifier fitness - initialized to a constant initial fitness value
        self.accuracy = 0.0  # Classifier accuracy - Accuracy calculated using only instances in the dataset which this rule matched.
        self.numerosity = 1  # The number of rule copies stored in the population.  (Indirectly stored as incremented numerosity)
        self.averageActionSetSize = 1.0  # A parameter used in deletion which reflects the size of match sets within this rule has been included.
        self.deletionVote = 0.0  # The current deletion weight for this classifier.

        # Experience Management ---------------------------------------------
        self.gaTimestamp = None  # Time since rule last in a match set.
        self.initTimestamp = None  # Iteration in which the rule first appeared.

        # Classifier Accuracy Tracking --------------------------------------
        self.ga_count = 0
        self.action_cnt = 0  # The total number of times this classifier was chosen in action set

        if isinstance(b, list):
            self.classifierCovering(a, b, c)
        elif isinstance(a, Classifier):
            self.classifierCopy(a)
        elif isinstance(a, list) and b is None:
            self.rebootClassifier(a)
        else:
            print("Classifier: Error building classifier.")

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # CLASSIFIER CONSTRUCTION METHODS
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def classifierCovering(self, iteration, state, action):
        """ Makes a new classifier when the covering mechanism is triggered.  The new classifier will match the current training instance.
        Covering will NOT produce a default rule (i.e. a rule with a completely general condition). """
        # Initialize new classifier parameters----------
        self.gaTimestamp = iteration
        self.initTimestamp = iteration
        if action is not None:
            self.action = action
        else:
            self.action = random.choice(cons.env.format_data.phenotypeList)
        # -------------------------------------------------------
        # GENERATE MATCHING CONDITION
        # -------------------------------------------------------
        for att in range(cons.env.format_data.numAttributes):
            if random.random() < cons.p_spec and state[att] != cons.labelMissingData:
                self.specifiedAttributes.append(att)
                self.condition.append(state[att])

    def classifierCopy(self, old_cl):
        """  Constructs an identical Classifier.  However, the experience of the copy is set to 0 and the numerosity
        is set to 1 since this is indeed a new individual in a population. Used by the genetic algorithm to generate
        offspring based on parent classifiers."""
        self.specifiedAttributes = old_cl.specifiedAttributes[:]
        self.condition = old_cl.condition[:]
        self.action = old_cl.action
        self.gaTimestamp = old_cl.gaTimestamp
        self.initTimestamp = old_cl.gaTimestamp
        self.averageActionSetSize = old_cl.averageActionSetSize
        self.prediction = old_cl.prediction
        self.error = old_cl.error
        self.fitness = old_cl.fitness / old_cl.numerosity

    def rebootClassifier(self, classifier_list):
        """ Rebuilds a saved classifier as part of the population Reboot """
        numAttributes = cons.env.format_data.numAttributes
        for att in range(0, numAttributes):
            if classifier_list[att] != '#':  # Attribute in rule is not wild
                self.condition.append(classifier_list[att])
                self.specifiedAttributes.append(att)
        # -------------------------------------------------------
        # DISCRETE PHENOTYPE
        # -------------------------------------------------------
        if cons.env.format_data.discretephenotype:
            self.action = str(classifier_list[numAttributes])
        # -------------------------------------------------------
        # CONTINUOUS PHENOTYPE
        # -------------------------------------------------------
        else:
            self.action = classifier_list[numAttributes].split(';')
            for i in range(2):
                self.action[i] = float(self.action[i])

        self.prediction = float(classifier_list[numAttributes + 1])
        self.error = float(classifier_list[numAttributes + 2])
        self.fitness = float(classifier_list[numAttributes + 3])
        self.numerosity = int(classifier_list[numAttributes + 4])
        self.ga_count = float(classifier_list[numAttributes + 5])
        self.averageActionSetSize = float(classifier_list[numAttributes + 6])
        self.gaTimestamp = int(classifier_list[numAttributes + 7])
        self.initTimestamp = int(classifier_list[numAttributes + 8])

        self.deletionVote = float(classifier_list[numAttributes + 10])
        self.action_cnt = int(classifier_list[numAttributes + 11])

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # MATCHING
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def doesMatch(self, state):
        """ Returns if the classifier matches in the current situation. """
        for i in range(len(self.condition)):
            state_val = state[self.specifiedAttributes[i]]
            if state_val == self.condition[i] or state_val == cons.labelMissingData:
                pass
            else:
                return False
        return True

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # GENETIC ALGORITHM MECHANISMS
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def uniformCrossover(self, cl):
        """ Applies uniform crossover and returns if the classifiers changed. Handles both discrete and continuous attributes.
        #SWARTZ: self. is where for the better attributes are more likely to be specified
        #DEVITO: cl. is where less useful attribute are more likely to be specified
        """
        self_specifiedAttributes = self.specifiedAttributes[:]
        cl_specifiedAttributes = cl.specifiedAttributes[:]
        probability = 0.5  # Equal probability for attribute alleles to be exchanged.

        # Make list of attribute references appearing in at least one of the parents.-----------------------------
        combined_specified_atts = []
        for i in self_specifiedAttributes:
            combined_specified_atts.append(i)
        for i in cl_specifiedAttributes:
            if i not in combined_specified_atts:
                combined_specified_atts.append(i)
            else:  # Attribute specified in both parents, and the attribute is discrete (then no reason to cross over)
                combined_specified_atts.remove(i)
        combined_specified_atts.sort()
        # --------------------------------------------------------------------------------------------------------
        changed = False
        for att in combined_specified_atts:  # Each condition specifies different attributes, so we need to go through all attributes in the dataset.
            # -----------------------------
            if att in self_specifiedAttributes and random.random() > probability:
                i = self.specifiedAttributes.index(
                    att)  # reference to the position of the attribute in the rule representation
                cl.condition.append(self.condition.pop(i))  # Take attribute from self and add to cl
                cl.specifiedAttributes.append(att)
                self.specifiedAttributes.remove(att)
                changed = True  # Remove att from self and add to cl
            if att in cl_specifiedAttributes and random.random() < probability:
                i = cl.specifiedAttributes.index(
                    att)  # reference to the position of the attribute in the rule representation
                self.condition.append(cl.condition.pop(i))  # Take attribute from self and add to cl
                self.specifiedAttributes.append(att)
                cl.specifiedAttributes.remove(att)
                changed = True  # Remove att from cl and add to self.

        tmp_list1 = self_specifiedAttributes[:]
        tmp_list2 = cl.specifiedAttributes[:]
        tmp_list1.sort()
        tmp_list2.sort()
        if changed and (tmp_list1 == tmp_list2):
            changed = False

        if self.action != cl.action and random.random() > probability:
            # Switch phenotypes of 2 classifiers if GA is run in match set
            temp = self.action
            self.action = cl.action
            cl.action = temp
            changed = True
        return changed

    def twoPointCrossover(self, cl):
        """ Applies two point crossover and returns if the classifiers changed. Handles merely discrete attributes and phenotypes """
        points = [None, None]
        changed = False
        points[0] = int(random.random() * (cons.env.format_data.numAttributes))
        points[1] = int(random.random() * (cons.env.format_data.numAttributes))
        if points[0] > points[1]:
            temp_point = points[0]
            points[0] = points[1]
            points[1] = temp_point
        self_specifiedAttributes = self.specifiedAttributes[:]
        cl_specifiedAttributes = cl.specifiedAttributes[:]
        for i in range(points[0], points[1] + 1):
            if i in self_specifiedAttributes:
                if i not in cl_specifiedAttributes:
                    index = self.specifiedAttributes.index(i)
                    cl.condition.append(self.condition.pop(index))
                    cl.specifiedAttributes.append(i)
                    self.specifiedAttributes.remove(i)
                    changed = True  # Remove att from self and add to cl
            elif i in cl_specifiedAttributes:
                index = cl.specifiedAttributes.index(
                    i)  # reference to the position of the attribute in the rule representation
                self.condition.append(cl.condition.pop(index))  # Take attribute from self and add to cl
                self.specifiedAttributes.append(i)
                cl.specifiedAttributes.remove(i)
                changed = True
        return changed

    def actionCrossover(self, cl):
        """ Crossover a continuous phenotype """
        changed = False
        if self.action[0] == cl.action[0] and self.action[1] == cl.action[1]:
            return changed
        else:
            tmp_key = random.random() < 0.5  # Make random choice between 4 scenarios, Swap minimums, Swap maximums, Children preserve parent phenotypes.
            if tmp_key:  # Swap minimum
                temp = self.action[0]
                self.action[0] = cl.action[0]
                cl.action[0] = temp
                changed = True
            elif tmp_key:  # Swap maximum
                temp = self.action[1]
                self.action[1] = cl.action[1]
                cl.action[1] = temp
                changed = True

        return changed

    def Mutation(self, state):
        """ Mutates the condition of the classifier. Also handles phenotype mutation. This is a niche mutation, which means that the resulting classifier will still match the current instance.  """
        changed = False
        # -------------------------------------------------------
        # MUTATE CONDITION
        # -------------------------------------------------------
        for att in range(
                cons.env.format_data.numAttributes):  # Each condition specifies different attributes, so we need to go through all attributes in the dataset.
            if random.random() < cons.mu and state[att] != cons.labelMissingData:
                # MUTATION--------------------------------------------------------------------------------------------------------------
                if att not in self.specifiedAttributes:  # Attribute not yet specified
                    self.specifiedAttributes.append(att)
                    self.condition.append(state[att])  # buildMatch handles both discrete and continuous attributes
                    changed = True
                elif att in self.specifiedAttributes:  # Attribute already specified
                    i = self.specifiedAttributes.index(
                        att)  # reference to the position of the attribute in the rule representation
                    self.specifiedAttributes.remove(att)
                    self.condition.pop(i)  # buildMatch handles both discrete and continuous attributes
                    changed = True
                # -------------------------------------------------------
                # NO MUTATION OCCURS
                # -------------------------------------------------------
                else:
                    pass
        # -------------------------------------------------------
        # MUTATE PHENOTYPE
        # -------------------------------------------------------
        if random.random() < cons.mu:
            phenotypeList = cons.env.format_data.phenotypeList[:]
            phenotypeList.remove(self.action)
            self.action = random.choice(phenotypeList)
            changed = True
        return changed

    def discreteActionMutation(self):
        """ Mutate this rule's discrete phenotype. """
        changed = False
        if random.random() < cons.mu:
            phenotypeList = cons.env.format_data.phenotypeList[:]
            phenotypeList.remove(self.action)
            self.action = random.choice(phenotypeList)
            changed = True
        return changed

    def continuousActionMutation(self, phenotype):
        """ Mutate this rule's continuous phenotype. """
        changed = False
        if random.random() < cons.mu:  # Mutate continuous phenotype
            action_range = self.action[1] - self.action[0]
            mutate_range = random.random() * 0.5 * action_range
            tmp_key = random.randint(0,
                                     2)  # Make random choice between 3 scenarios, mutate minimums, mutate maximums, mutate both
            if tmp_key == 0:  # Mutate minimum
                if random.random() > 0.5 or self.action[
                    0] + mutate_range <= phenotype:  # Checks that mutated range still contains current phenotype
                    self.action[0] += mutate_range
                else:  # Subtract
                    self.action[0] -= mutate_range
                changed = True
            elif tmp_key == 1:  # Mutate maximum
                if random.random() > 0.5 or self.action[
                    1] - mutate_range >= phenotype:  # Checks that mutated range still contains current phenotype
                    self.action[1] -= mutate_range
                else:  # Subtract
                    self.action[1] += mutate_range
                changed = True
            else:  # mutate both
                if random.random() > 0.5 or self.action[
                    0] + mutate_range <= phenotype:  # Checks that mutated range still contains current phenotype
                    self.action[0] += mutate_range
                else:  # Subtract
                    self.action[0] -= mutate_range
                if random.random() > 0.5 or self.action[
                    1] - mutate_range >= phenotype:  # Checks that mutated range still contains current phenotype
                    self.action[1] -= mutate_range
                else:  # Subtract
                    self.action[1] += mutate_range
                changed = True

            # Repair range - such that min specified first, and max second.
            self.action.sort()
        # ---------------------------------------------------------------------
        return changed

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # SUBSUMPTION METHODS
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def subsumes(self, cl):
        """ Returns if the classifier (self) subsumes cl """
        if cl.action == self.action and self.isPossibleSubsumer() and self.isMoreGeneral(cl):
            return True
        return False

    def isPossibleSubsumer(self):
        """ Returns if the classifier (self) is a possible subsumer. A classifier must be as or more accurate than the classifier it is trying to subsume.  """
        if self.action_cnt > cons.theta_sub and self.error < cons.err_sub:  # self.prediction < cons.err_sub: (why does it work?)
            return True
        return False

    def isMoreGeneral(self, cl):
        """ Returns if the classifier (self) is more general than cl. Check that all attributes specified in self are also specified in cl. """
        if len(self.specifiedAttributes) >= len(
                cl.specifiedAttributes):  # and self.action != cl.action and self.prediction < cl.prediction and self.error > cl.error:
            return False
        for i in range(len(self.specifiedAttributes)):  # Check each attribute specified in self.condition
            if self.specifiedAttributes[i] not in cl.specifiedAttributes:
                return False
        return True

    def compactSubsumes(self, cl):
        """ Returns whether the classifier (self) subsumes cl (in compacting). """
        if len(self.specifiedAttributes) > len(
                cl.specifiedAttributes):  # and self.action != cl.action and self.prediction < cl.prediction and self.error > cl.error:
            return False
        if cl.action == self.action:
            for i in range(len(self.specifiedAttributes)):
                if self.specifiedAttributes[i] not in cl.specifiedAttributes:
                    return False
                else:
                    j = cl.specifiedAttributes.index(self.specifiedAttributes[i])
                    if self.condition[i] != cl.condition[j]:
                        return False
            return True
        return False

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # DELETION METHOD
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def getDelProb(self, avg_fitness):
        """  Returns the vote for deletion of the classifier. """
        self.deletionVote = self.averageActionSetSize * self.numerosity
        if self.action_cnt > cons.theta_del and self.fitness / self.numerosity < cons.delta * avg_fitness:
            if self.fitness != 0.0:
                self.deletionVote *= avg_fitness * self.numerosity / self.fitness  # same as avg_fitness / ( self.fitness/self.numerosity )
            else:
                self.deletionVote *= avg_fitness * self.numerosity / cons.init_fit  # same as avg_fitness / ( cons.init_fit/self.numerosity )
        return self.deletionVote

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # OTHER METHODS
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def equals(self, cl, niched=True):
        """ Returns if the two classifiers are identical in condition and phenotype. This works for discrete or continuous attributes or phenotypes. """
        if cl.action == self.action and len(cl.specifiedAttributes) == len(
                self.specifiedAttributes):  # Is phenotype the same and are the same number of attributes specified - quick equality check first.
            cl_atts = sorted(cl.specifiedAttributes)
            self_atts = sorted(self.specifiedAttributes)
            if cl_atts == self_atts:
                if not niched:  # compare condition if comparison between classifiers in different niches.
                    for i in range(len(cl.specifiedAttributes)):
                        tmp_index = self.specifiedAttributes.index(cl.specifiedAttributes[i])
                        if cl.condition[i] == self.condition[tmp_index]:
                            pass
                        else:
                            return False
                return True
        return False

    def updateXCSParameters(self, reward):
        """ Update the XCS classifier parameters: prediction payoff, prediction error and fitness. """
        payoff = reward
        if self.action_cnt >= 1.0 / cons.beta:
            self.error += cons.beta * (abs(payoff - self.prediction) - self.error)
            self.prediction += cons.beta * (payoff - self.prediction)
        else:
            self.error = (self.error * (self.action_cnt - 1) + abs(payoff - self.prediction)) / self.action_cnt
            self.prediction = (self.prediction * (self.action_cnt - 1) + payoff) / self.action_cnt
        if self.error <= cons.offset_epsilon:
            self.accuracy = 1.0
        else:
            self.accuracy = cons.alpha * ((self.error / cons.offset_epsilon) ** (
                -cons.nu))  # math.pow( cons.alpha, ( self.error - cons.offset_epsilon ) / cons.offset_epsilon )

    def updateFitness(self, local_accuracy):
        """ Update fitness of classifier/rule. """
        self.fitness += cons.beta * (local_accuracy - self.fitness)

    def updateActionSetSize(self, actionset_size):
        """  Updates the average action set size. """
        if self.action_cnt >= 1.0 / cons.beta:
            self.averageActionSetSize += cons.beta * (actionset_size - self.averageActionSetSize)
        else:
            self.averageActionSetSize = (self.averageActionSetSize * (self.action_cnt - 1) + actionset_size) / float(
                self.action_cnt)

    def updateExperience(self):
        """ Increases the experience of the classifier by one. Once an epoch has completed, rule accuracy can't change."""
        self.matchCount += 1

    def updateActionExp(self):
        """ Increases the experience of the classifier by one. Once an epoch has completed, rule accuracy can't change."""
        self.action_cnt += 1

    def updateGACount(self):
        """ Increment number of times the classifier is selected in GA by one, for statistics. """
        self.ga_count += 1

    def updateNumerosity(self, num):
        """ Updates the numberosity of the classifier.  Notice that 'num' can be negative! """
        self.numerosity += num

    def updateTimeStamp(self, ts):
        """ Sets the time stamp of the classifier. """
        self.gaTimestamp = ts

    def setPrediction(self, pred):
        """ Sets the accuracy of the classifier """
        self.prediction = pred

    def setError(self, err):
        """ Sets the accuracy of the classifier """
        self.error = err

    def setAccuracy(self, acc):
        """ Sets the accuracy of the classifier """
        self.accuracy = acc

    def setFitness(self, fit):
        """  Sets the fitness of the classifier. """
        self.fitness = fit

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # PRINT CLASSIFIER FOR POPULATION OUTPUT FILE
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def printClassifier(self):
        """ Formats and returns an output string describing this classifier. """
        classifier_info = ""
        for att in range(cons.env.format_data.numAttributes):
            if att in self.specifiedAttributes:  # If the attribute was specified in the rule
                i = self.specifiedAttributes.index(att)
                classifier_info += str(self.condition[i]) + "\t"
            else:  # Attribute is wild.
                classifier_info += '#' + "\t"
        # -------------------------------------------------------------------------------
        specificity = len(self.condition) / float(cons.env.format_data.numAttributes)

        if cons.env.format_data.discretephenotype:
            classifier_info += '{0:>5}'.format(self.action) + "\t"
        else:
            classifier_info += '{0:>3}'.format(self.action[0]) + ';' + '{0:>3}'.format(self.action[1]) + "\t"
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        classifier_info += '{0:>6.1f}'.format(self.prediction) + '\t' + '{0:>6.1f}'.format(
            self.error) + '\t' + '{0:>5.3f}'.format(self.fitness) + "\t" + '{0:>4}'.format(self.numerosity) + '\t'
        classifier_info += '{0:>6.1f}'.format(self.averageActionSetSize) + '\t' + '{0:>8}'.format(
            self.gaTimestamp) + '\t' + '{0:>8}'.format(self.initTimestamp) + '\t' + '{:.2f}'.format(specificity) + '\t'
        classifier_info += '{:>10.1f}'.format(self.deletionVote) + '\t' + '{:>8}'.format(self.action_cnt) + '\n'

        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        return classifier_info
