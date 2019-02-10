# Import Required Modules--------------
from XCS.XCS_Constants import *


# -------------------------------------

class OutputFileManager:
    def writePopStats(self, outFileName, train_eval, test_eval, iteration, pop, correct):
        """ Makes output text file which includes all of the evaluation statistics for a complete analysis of all training and testing data on the current XCS rule population. """
        try:
            pop_stats_out = open(outFileName + '_' + str(iteration) + '_PopStats.txt', 'w')
        except Exception as inst:
            print(type(inst))
            print(inst.args)
            print(inst)
            print('cannot open', outFileName + '_' + str(iteration) + '_PopStats.txt')
            raise
        else:
            print("Writing Population Statistical Summary File...")

        # Evaluation of pop
        pop_stats_out.write(
            "Performance Statistics:------------------------------------------------------------------------\n")
        pop_stats_out.write("Training Accuracy\tTesting Accuracy\tTraining Coverage\tTesting Coverage\n")

        if cons.testFile != 'None':
            pop_stats_out.write(str(train_eval[0]) + "\t")
            pop_stats_out.write(str(test_eval[0]) + "\t")
            pop_stats_out.write(str(train_eval[1]) + "\t")
            pop_stats_out.write(str(test_eval[1]) + "\n\n")
        elif cons.trainFile != 'None':
            pop_stats_out.write(str(train_eval[0]) + "\t")
            pop_stats_out.write("NA\t")
            pop_stats_out.write(str(train_eval[1]) + "\t")
            pop_stats_out.write("NA\n\n")
        else:
            pop_stats_out.write("NA\t")
            pop_stats_out.write("NA\t")
            pop_stats_out.write("NA\t")
            pop_stats_out.write("NA\n\n")

        pop_stats_out.write(
            "Population Characterization:------------------------------------------------------------------------\n")
        pop_stats_out.write("MacroPopSize\tMicroPopSize\tGenerality\n")
        pop_stats_out.write(
            str(len(pop.population)) + "\t" + str(pop.micro_size) + "\t" + str(pop.mean_generality) + "\n\n")

        pop_stats_out.write("SpecificitySum:------------------------------------------------------------------------\n")
        headers = cons.env.format_data.trainHeaderList  # preserve order of original dataset

        for i in range(len(headers)):
            if i < len(headers) - 1:
                pop_stats_out.write(str(headers[i]) + "\t")
            else:
                pop_stats_out.write(str(headers[i]) + "\n")

        # Prints out the Specification Sum for each attribute
        for i in range(len(pop.attribute_spec_list)):
            if i < len(pop.attribute_spec_list) - 1:
                pop_stats_out.write(str(pop.attribute_spec_list[i]) + "\t")
            else:
                pop_stats_out.write(str(pop.attribute_spec_list[i]) + "\n")

        pop_stats_out.write("\nAccuracySum:------------------------------------------------------------------------\n")
        for i in range(len(headers)):
            if i < len(headers) - 1:
                pop_stats_out.write(str(headers[i]) + "\t")
            else:
                pop_stats_out.write(str(headers[i]) + "\n")

        # Prints out the Accuracy Weighted Specification Count for each attribute
        for i in range(len(pop.attribute_acc_list)):
            if i < len(pop.attribute_acc_list) - 1:
                pop_stats_out.write(str(pop.attribute_acc_list[i]) + "\t")
            else:
                pop_stats_out.write(str(pop.attribute_acc_list[i]) + "\n")

        # Time Track ---------------------------------------------------------------------------------------------------------
        pop_stats_out.write(
            "\nRun Time(in minutes):------------------------------------------------------------------------\n")
        pop_stats_out.write(cons.timer.reportTimes())
        pop_stats_out.write(
            "\nCorrectTrackerSave:------------------------------------------------------------------------\n")
        for i in range(len(correct)):
            pop_stats_out.write(str(correct[i]) + "\t")

        pop_stats_out.close()

    def writePop(self, outFileName, iteration, pop):
        """ Writes a tab delimited text file outputting the entire evolved rule population, including conditions, phenotypes, and all rule parameters. """
        try:
            rule_pop_out = open(outFileName + '_' + str(iteration) + '_RulePop.txt', 'w')
        except Exception as inst:
            print(type(inst))
            print(inst.args)
            print(inst)
            print('cannot open', outFileName + '_' + str(iteration) + '_RulePop.txt')
            raise
        else:
            print("Writing Population as Data File...")

        # Write Header-----------------------------------------------------------------------------------------------------------------------------------------------
        headers = cons.env.format_data.trainHeaderList
        for i in range(len(headers)):
            rule_pop_out.write(str(headers[i]) + "\t")
        rule_pop_out.write(
            "Action\tPredic\t Error\tFitne\t Num\tASSize\tGATimeSt\tInitTime\tSpec\tDeleteProb\t   ASCnt\n")

        # sort classifiers based on numerosity-----------------------------------------------------------------------------------------------------------------------
        sorted_pop = sorted(pop.population, key=lambda x: x.numerosity, reverse=True)
        # Write each classifier--------------------------------------------------------------------------------------------------------------------------------------
        for cl in sorted_pop:
            rule_pop_out.write(str(cl.printClassifier()))

        rule_pop_out.close()
