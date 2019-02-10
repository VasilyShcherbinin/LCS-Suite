# Import Required Modules---------------------
import random

from XCS.XCS_Classifier import Classifier
from XCS.XCS_Constants import *


# --------------------------------------------

class ClassifierSet:
    def __init__(self, a=None):
        """ Overloaded initialization: Handles creation of a new population or a rebooted population (i.e. a previously saved population). """
        # Major Parameters
        self.population = []  # List of classifiers/rules
        self.matchSet = []  # List of references to rules in population that match
        self.actionSet = []  # List of references to rules in population that match and has action with highest prediction payoff
        self.micro_size = 0  # Tracks the current micro population size, i.e. the population size which takes rule numerosity into account.

        # Evaluation Parameters-------------------------------
        self.mean_generality = 0.0
        self.attribute_spec_list = []
        self.attribute_acc_list = []
        self.avg_action_range = 0.0

        # Set Constructors-------------------------------------
        if a is None:
            self.initPopulation()  # Initialize a new population
        elif isinstance(a, str):
            self.rebootPop(a)  # Initialize a population based on an existing saved rule population
        else:
            print("ClassifierSet: Error building population.")

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # POPULATION CONSTRUCTOR METHODS
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def initPopulation(self):
        """ Initializes the rule population """
        self.population = []

    def rebootPop(self, remakeFile):
        """ Remakes a previously evolved population from a saved text file. """
        print("Rebooting the following population: " + str(remakeFile) + "_RulePop.txt")
        # *******************Initial file handling**********************************************************
        dataset_list = []
        try:
            f = open(remakeFile + "_RulePop.txt", 'r')
        except Exception as inst:
            print(type(inst))
            print(inst.args)
            print(inst)
            print('cannot open', remakeFile + "_RulePop.txt")
            raise
        else:
            self.header_list = f.readline().rstrip('\n').split('\t')  # strip off first row
            for line in f:
                line_list = line.strip('\n').split('\t')
                dataset_list.append(line_list)
            f.close()

        # **************************************************************************************************
        for each in dataset_list:
            cl = Classifier(each)
            self.population.append(cl)
            numerosity_ref = cons.env.format_data.numAttributes + 3
            self.micro_size += int(each[numerosity_ref])
        print("Rebooted Rule Population has " + str(len(self.population)) + " Macro Pop Size.")

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # CLASSIFIER SET CONSTRUCTOR METHODS
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def generateMatchSet(self, state, iteration):
        """ Constructs a match set from the population. Covering is initiated if the match set is empty or total prediction of rules in match set is too low. """
        # Initial values
        match_set = []
        self.current_instance = state
        # -------------------------------------------------------
        # MATCHING
        # -------------------------------------------------------
        cons.timer.startTimeMatching()
        for i in range(len(self.population)):  # Go through the population
            cl = self.population[i]  # One classifier at a time
            if cl.doesMatch(state):  # Check for match
                self.matchSet.append(i)  # If match - add classifier to match set
                if cl.action not in match_set:
                    match_set.append(cl.action)
        cons.timer.stopTimeMatching()
        # -------------------------------------------------------
        # COVERING
        # -------------------------------------------------------
        if len(match_set) < cons.theta_mna:
            missing_actions = [a for a in cons.env.format_data.phenotypeList if a not in match_set]
            for action in missing_actions:
                new_cl = Classifier(iteration, state, action)
                self.addClassifierToPopulation(new_cl)
                self.matchSet.append(len(self.population) - 1)  # Add created classifier to match set
                match_set.append(new_cl.action)
            if len(match_set) >= cons.theta_mna:
                self.deletion()
                match_set = []
                for i in self.matchSet:
                    if self.population[i].action not in match_set:
                        match_set.append(self.population[i].action)

    def generateActionSet(self, selected_action):
        """ Constructs a correct set out of the given match set. """
        for ref in self.matchSet:
            if self.population[ref].action == selected_action:
                self.actionSet.append(ref)

    def makeEvalMatchSet(self, state):
        """ Constructs a match set for evaluation purposes which does not activate either covering or deletion. """
        for i in range(len(self.population)):  # Go through the population
            cl = self.population[i]  # A single classifier
            if cl.doesMatch(state):  # Check for match
                self.matchSet.append(i)  # Add classifier to match set

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # CLASSIFIER DELETION METHODS
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def deletion(self):
        """ Returns the population size back to the maximum set by the user by deleting rules. """
        cons.timer.startTimeDeletion()
        while self.micro_size > cons.N:
            self.deleteFromPopulation()
        cons.timer.stopTimeDeletion()

    def deleteFromPopulation(self):
        """ Deletes one classifier in the population.  The classifier that will be deleted is chosen by roulette wheel selection
        considering the deletion vote. Returns the macro-classifier which got decreased by one micro-classifier. """
        mean_fitness = self.getPopFitnessSum() / float(self.micro_size)
        # Calculate total wheel size------------------------------
        set_size = len(self.population)
        vote_sum = 0.0
        vote_list = [0.0] * set_size
        for i in range(set_size):
            cl = self.population[i]
            vote = cl.getDelProb(mean_fitness)
            vote_sum += vote
            vote_list[i] = vote
        # --------------------------------------------------------
        choice_point = vote_sum * random.random()  # Determine the choice point
        new_sum = 0.0
        for i in range(len(vote_list)):
            new_sum = new_sum + vote_list[i]
            if new_sum > choice_point:  # Select classifier for deletion
                # Delete classifier----------------------------------
                cl = self.population[i]
                cl.updateNumerosity(-1)
                self.micro_size -= 1
                if cl.numerosity < 1:  # When all micro-classifiers for a given classifier have been depleted.
                    self.removeMacroClassifier(i)
                    if self.matchSet != []:
                        self.deleteFromMatchSet(i)
                        self.deleteFromActionSet(i)
                return
        print("ClassifierSet: No eligible rules found for deletion in deleteFromPopulation.")
        return

    def removeMacroClassifier(self, ref):
        """ Removes the specified (macro-) classifier from the population. """
        self.population.pop(ref)

    def deleteFromMatchSet(self, cl_ref):
        """ Delete reference to classifier in population, contained in self.match_set."""
        if cl_ref in self.matchSet:
            self.matchSet.remove(cl_ref)
        # Update match set reference list--------
        for j in range(len(self.matchSet)):
            ref = self.matchSet[j]
            if ref > cl_ref:
                self.matchSet[j] -= 1

    def deleteFromActionSet(self, cl_ref):
        """ Delete reference to classifier in population, contained in self.action_set."""
        if cl_ref in self.actionSet:
            self.actionSet.remove(cl_ref)

        # Update action set reference list--------
        for j in range(len(self.actionSet)):
            ref = self.actionSet[j]
            if ref > cl_ref:
                self.actionSet[j] -= 1

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # GENETIC ALGORITHM
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def runGA(self, iteration, state):
        """ The genetic discovery mechanism in XCS is controlled here. """
        # -------------------------------------------------------
        # GA RUN REQUIREMENT
        # -------------------------------------------------------
        if (
                iteration - self.getIterStampAverage()) < cons.theta_GA:  # Does the action set meet the requirements for activating the GA?
            self.clearSets()
            return

        self.setIterStamps(
            iteration)  # Updates the iteration time stamp for all rules in the match set (which the GA opperates in).
        changed = False

        # -------------------------------------------------------
        # SELECT PARENTS - Niche GA - selects parents from the match set
        # -------------------------------------------------------
        cons.timer.startTimeSelection()
        if cons.selectionMethod == "roulette":
            selected_list = self.selectClassifierRW()
            clP1 = selected_list[0]
            clP2 = selected_list[1]
        elif cons.selectionMethod == "tournament":
            selected_list = self.selectClassifierT()
            clP1 = selected_list[0]
            clP2 = selected_list[1]
        cons.timer.stopTimeSelection()

        # clP1.updateGACount()
        # clP2.updateGACount()
        # -------------------------------------------------------
        # INITIALIZE OFFSPRING
        # -------------------------------------------------------
        cl1 = Classifier(clP1, iteration)
        if clP2 is None:
            cl2 = Classifier(clP1, iteration)
        else:
            cl2 = Classifier(clP2, iteration)
        # -------------------------------------------------------
        # CROSSOVER OPERATOR - Uniform Crossover Implemented (i.e. all attributes have equal probability of crossing over between two parents)
        # -------------------------------------------------------
        if not cl1.equals(cl2) and random.random() < cons.chi:
            if cons.crossoverMethod == 'uniform':
                changed = cl1.uniformCrossover(cl2)
            elif cons.crossoverMethod == 'twopoint':
                changed = cl1.twoPointCrossover(cl2)
        # -------------------------------------------------------
        # MUTATION OPERATOR
        # -------------------------------------------------------
        mutate_change1 = cl1.Mutation(state)
        mutate_change2 = cl2.Mutation(state)
        # -------------------------------------------------------
        # INITIALIZE KEY OFFSPRING PARAMETERS
        # -------------------------------------------------------
        if changed or mutate_change1 or mutate_change2:
            cl1.setPrediction((cl1.prediction + cl2.prediction) / 2)
            cl1.setError((cl1.error + cl2.error) / 2.0)
            cl1.setFitness(cons.fitness_reduction * (cl1.fitness + cl2.fitness) / 2.0)
            cl2.setPrediction(cl1.prediction)
            cl2.setError(cl1.error)
            cl2.setFitness(cl1.fitness)
        # -------------------------------------------------------
        # ADD OFFSPRING TO POPULATION
        # -------------------------------------------------------
        self.insertDiscoveredClassifiers(cl1, cl2, clP1, clP2)  # Subsumption
        self.clearSets()
        self.deletion()

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # SELECTION METHODS
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def selectClassifierRW(self):
        """ Selects parents using roulette wheel selection according to the fitness of the classifiers. """
        # Prepare for actionSet set or 'niche' selection.
        selected_list = [None, None]
        count = 0  # Pick two parents
        set_list = self.actionSet[:]
        # -----------------------------------------------
        while count < 2:
            fit_sum = self.getFitnessSum(set_list)

            choice_point = random.random() * fit_sum
            i = 0
            sum_cl = self.population[set_list[i]].fitness
            while choice_point > sum_cl:
                i = i + 1
                sum_cl += self.population[set_list[i]].fitness

            selected_list[count] = self.population[set_list[i]]
            if cons.distinctParents and len(set_list) > 1:
                set_list.pop(i)
            count += 1
        # -----------------------------------------------
        return selected_list

    def selectClassifierT(self):
        """  Selects parents using tournament selection according to the fitness of the classifiers. """
        selected_list = [None, None]
        count = 0
        set_list = self.actionSet[:]  # actionSet set is a list of reference IDs
        # -----------------------------------------------
        while count < 2:
            tournament_size = int(len(set_list) * cons.theta_sel)
            if tournament_size < 1:
                tournament_size = 1
            post_list = random.sample(set_list, tournament_size)
            highest_fitness = 0
            best_cl = post_list[0]
            for j in post_list:
                if self.population[j].fitness > highest_fitness:
                    highest_fitness = self.population[j].fitness
                    best_cl = j
            selected_list[count] = self.population[best_cl]
            count += 1
            if cons.distinctParents and len(set_list) > 1:
                set_list.remove(best_cl)

        return selected_list

    # def selectClassifierUsingIqbalTournamentSel(self):
    #     winnerSet = []
    #     fitness = -1.0
    #     select_tolerance = 0
    #
    #     # there must be at least one classifier in the set
    #     # if classifierset==None or len(classifierset)==0:
    #     #    print "in selectClassifierUsingIqbalTournamentSel classifierset mustn't be None"
    #
    #     # only one classifier in set
    #     if len(self.action_set) == 1:
    #         return self.action_set[0]
    #
    #     # tournament with fixed size
    #     # tournament selection with the tournament size approx. equal to tournamentSize setsum
    #     while len(winnerSet) == 0:
    #         for i in self.action_set:
    #             # if his fitness is worse then do not bother
    #             if len(winnerSet) == 0 or (fitness - select_tolerance) <= self.pop_set[i].fitness/self.pop_set[i].numerosity:
    #                 for j in range(0, self.pop_set[i].numerosity):
    #                     if random.random() < cons.theta_sel:
    #                         if len(winnerSet) == 0:
    #                             # the first one
    #                             winnerSet.append(i)
    #                             fitness = self.pop_set[i].fitness/self.pop_set[i].numerosity
    #
    #                         else:
    #                             if (fitness + select_tolerance) > self.pop_set[i].fitness/self.pop_set[i].numerosity:
    #                                 winnerSet.append(i)
    #
    #                             else:
    #                                 winnerSet = []
    #                                 winnerSet.append(i)
    #                                 fitness = self.pop_set[i].fitness/self.pop_set[i].numerosity
    #                         break
    #
    #         # print winnerSet
    #     if len(winnerSet) > 1:
    #         size = random.randint(0, len(winnerSet) - 1)
    #         return winnerSet[size]
    #     return winnerSet[0]

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # SUBSUMPTION METHODS
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def subsumeClassifier(self, cl=None, cl1P=None, cl2P=None, num_copy=1):
        """ Tries to subsume a classifier in the parents. If no subsumption is possible it tries to subsume it in the current set. """
        if cl1P != None and cl1P.subsumes(cl):
            self.micro_size += num_copy
            cl1P.updateNumerosity(num_copy)
        elif cl2P != None and cl2P.subsumes(cl):
            self.micro_size += num_copy
            cl2P.updateNumerosity(num_copy)
        else:
            # self.addClassifierToPopulation(cl)
            self.subsumeClassifier2(cl, num_copy)  # Try to subsume in the match set.

    def subsumeClassifier2(self, cl, num_copy=1):
        """ Tries to subsume a classifier in the match set. If no subsumption is possible the classifier is simply added to the population considering
        the possibility that there exists an identical classifier. """
        choices = []
        for ref in self.matchSet:
            if self.population[ref].subsumes(cl):
                choices.append(ref)

        if len(choices) > 0:  # Randomly pick one classifier to be subsumer
            choicep = int(random.random() * len(choices))
            self.population[choices[choicep]].updateNumerosity(num_copy)
            self.micro_size += num_copy
            return

        self.addClassifierToPopulation(cl,
                                       num_copy=num_copy)  # If no subsumer was found, check for identical classifier, if not then add the classifier to the population

    def doActionSetSubsumption(self):
        """ Executes match set subsumption.  The match set subsumption looks for the most general subsumer classifier in the match set
        and subsumes all classifiers that are more specific than the selected one. """
        subsumer = None
        for ref in self.actionSet:
            cl = self.population[ref]
            if cl.isPossibleSubsumer():
                if subsumer is None or len(subsumer.specifiedAttributes) > len(cl.specifiedAttributes) or (
                (len(subsumer.specifiedAttributes) == len(cl.specifiedAttributes) and random.random() < 0.5)):
                    subsumer = cl

        if subsumer != None:  # If a subsumer was found, subsume all more specific classifiers in the match set
            i = 0
            while i < len(self.actionSet):
                ref = self.actionSet[i]
                if subsumer.isMoreGeneral(self.population[ref]):
                    subsumer.updateNumerosity(self.population[ref].numerosity)
                    self.removeMacroClassifier(ref)
                    self.deleteFromMatchSet(ref)
                    self.deleteFromActionSet(ref)
                    i = i - 1
                i = i + 1

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # OTHER KEY METHODS
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def addClassifierToPopulation(self, cl, covering=False, num_copy=1):
        """ Adds a classifier to the set and increases the microPopSize value accordingly."""
        old_cl = None
        if not covering:
            old_cl = self.getIdenticalClassifier(cl)
        if old_cl != None:  # found identical classifier
            old_cl.updateNumerosity(num_copy)
        else:
            self.population.append(cl)
        self.micro_size += num_copy

    def insertDiscoveredClassifiers(self, cl1, cl2, clP1, clP2):
        """ Inserts both discovered classifiers and activates GA subsumption if turned on. Also checks for default rule (i.e. rule with completely general condition) and
        prevents such rules from being added to the population, as it offers no predictive value within XCS. """
        # -------------------------------------------------------
        # SUBSUMPTION
        # -------------------------------------------------------
        if cons.doGASubsumption:
            cons.timer.startTimeSubsumption()
            if not cl1.equals(cl2):
                if len(cl1.specifiedAttributes) > 0:
                    self.subsumeClassifier(cl1, clP1, clP2)
                if len(cl2.specifiedAttributes) > 0:
                    self.subsumeClassifier(cl2, clP1, clP2)
            elif len(cl1.specifiedAttributes) > 0:
                cl1.numerosity = 2
                self.subsumeClassifier(cl1, clP1, clP2, num_copy=2)
            cons.timer.stopTimeSubsumption()
        # -------------------------------------------------------
        # ADD OFFSPRING TO POPULATION
        # -------------------------------------------------------
        else:  # Just add the new classifiers to the population.
            if not cl1.equals(cl2):
                if len(cl1.specifiedAttributes) > 0:
                    self.addClassifierToPopulation(cl1)  # False passed because this is not called for a covered rule.
                if len(cl2.specifiedAttributes) > 0:
                    self.addClassifierToPopulation(cl2)  # False passed because this is not called for a covered rule.
            elif len(cl1.specifiedAttributes) > 0:
                cl1.numerosity = 2
                self.addClassifierToPopulation(cl1, num_copy=2)

    def updateSets(self, reward):
        """ Updates all relevant parameters in the current match and match sets. """
        action_set_numer = 0
        for ref in self.actionSet:
            action_set_numer += self.population[ref].numerosity
        accuracy_sum = 0.0
        for ref in self.actionSet:
            self.population[ref].updateActionExp()
            self.population[ref].updateActionSetSize(action_set_numer)
            self.population[ref].updateXCSParameters(reward)
            accuracy_sum += self.population[ref].accuracy * self.population[ref].numerosity
        for ref in self.actionSet:
            self.population[ref].updateFitness(self.population[ref].accuracy * self.population[ref].numerosity / accuracy_sum)

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # OTHER METHODS
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def getIterStampAverage(self):
        """ Returns the average of the time stamps in the match set. """
        sum_cl = 0.0
        sum_numer = 0.0
        for ref in self.actionSet:
            sum_cl += self.population[ref].gaTimestamp * self.population[ref].numerosity
            sum_numer += self.population[ref].numerosity  # numerosity sum of match set
        return sum_cl / float(sum_numer)

    def setIterStamps(self, iteration):
        """ Sets the time stamp of all classifiers in the set to the current time. The current time
        is the number of exploration steps executed so far.  """
        for ref in self.actionSet:
            self.population[ref].updateTimeStamp(iteration)

    # def setPredictionArray(self,newPredictionArray):
    #    predictionArray = newPredictionArray

    def getFitnessSum(self, cl_set):
        """ Returns the sum of the fitnesses of all classifiers in the set. """
        sum_cl = 0.0
        for ref in cl_set:
            sum_cl += self.population[ref].fitness
        return sum_cl

    def getPopFitnessSum(self):
        """ Returns the sum of the fitnesses of all classifiers in the set. """
        sum_cl = 0.0
        for cl in self.population:
            sum_cl += cl.fitness * cl.numerosity
        return sum_cl

    def getIdenticalClassifier(self, new_cl):
        """ Looks for an identical classifier in the population. """
        for ref in self.matchSet:
            if new_cl.equals(self.population[ref]):
                return self.population[ref]
        return None

    def clearSets(self):
        """ Clears out references in the match and action sets for the next learning iteration. """
        self.matchSet = []
        self.actionSet = []

    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # EVALUTATION METHODS
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def runPopAveEval(self):
        """ Calculates some summary evaluations across the rule population including average generality. """
        generality_sum = 0
        # agedCount = 0
        for cl in self.population:
            generality_sum += (cons.env.format_data.numAttributes - len(cl.condition)) / float(
                cons.env.format_data.numAttributes) * cl.numerosity
        if self.micro_size == 0:
            self.mean_generality = 'NA'
        else:
            self.mean_generality = generality_sum / float(self.micro_size)

    def runAttGeneralitySum(self, is_eval_summary):
        """ Determine the population-wide frequency of attribute specification, and accuracy weighted specification.  Used in complete rule population evaluations. """
        if is_eval_summary:
            self.attribute_spec_list = [0] * cons.env.format_data.numAttributes
            self.attribute_acc_list = [0.0] * cons.env.format_data.numAttributes
            for cl in self.population:
                for ref in cl.specifiedAttributes:  # for each attRef
                    self.attribute_spec_list[ref] += cl.numerosity
                    self.attribute_acc_list[ref] += cl.numerosity * cl.accuracy

    def getPopTrack(self, accuracy, iteration, trackingFrequency):
        """ Returns a formated output string to be printed to the Learn Track output file. """
        population_info = str(iteration) + "\t" + str(len(self.population)) + "\t" + str(self.micro_size) + "\t" + str(
            accuracy) + "\t" + str(self.mean_generality) + "\t" + str(cons.timer.returnGlobalTimer()) + "\n"
        if cons.env.format_data.discretephenotype:  # discrete phenotype
            print(("Epoch: " + str(int(iteration / trackingFrequency)) + "\t Iteration: " + str(
                iteration) + "\t MacroPop: " + str(len(self.population)) + "\t MicroPop: " + str(
                self.micro_size) + "\t AccEstimate: " + str(accuracy) + "\t AveGen: " + str(
                self.mean_generality) + "\t Time: " + str(cons.timer.returnGlobalTimer())))
        # else: # continuous phenotype
        #     print(("Epoch: "+str(int(iteration/trackingFrequency))+"\t Iteration: " + str(iteration) + "\t MacroPop: " + str(len(self.pop_set))+ "\t MicroPop: " + str(self.micro_size) + "\t AccEstimate: " + str(accuracy) + "\t AveGen: " + str(self.mean_generality) + "\t PhenRange: " +str(self.avg_action_range) + "\t Time: " + str(cons.timer.returnGlobalTimer())))
        return population_info

    def parallelMatching(self, i):  # ( ( indices, condition, state, id ) ):
        """ used when multiprocessing is enabled. """
        if self.population[i].doesMatch(self.current_instance):
            return i
        return None

    def finalise(self, do_compact=False):
        """ Compact the population. """
        ### Remove inexperienced and inaccurate classifiers -------------------------
        i = 0
        while i < len(self.population):
            cl = self.population[i]
            if cl.action_cnt <= cons.theta_del or cl.error >= 0.0001:
                self.micro_size -= cl.numerosity
                self.population.pop(i)
            else:
                i += 1
        ### Subsume overspecific classifiers ----------------------------------------
        if do_compact:
            i = 0
            while i < len(self.population):
                j = i + 1
                while j < len(self.population):
                    if self.population[i].compactSubsumes(self.population[j]):
                        self.population[i].numerosity += self.population[j].numerosity
                        self.population.pop(j)
                        continue
                    elif self.population[j].compactSubsumes(self.population[i]):
                        self.population[j].numerosity += self.population[i].numerosity
                        self.population.pop(i)
                        i -= 1
                        break
                    j += 1
                i += 1
