# Import Required Modules---------------
import time


# --------------------------------------

class Timer:
    def __init__(self):
        # Global Time objects
        self.global_start_t = time.time()
        self.global_time = 0.0
        self.added_time = 0.0

        # Data Generating time variables
        self.start_generating_t = 0.0
        self.global_generating = 0.0

        # Checkpoint time variables
        self.checkpoint_time = 0.0

        # Match Time Variables
        self.start_matching_t = 0.0
        self.global_matching = 0.0

        # Deletion Time Variables
        self.start_deletion_t = 0.0
        self.global_deletion = 0.0

        # Subsumption Time Variables
        self.start_subsumption_t = 0.0
        self.global_subsumption = 0.0

        # Selection Time Variables
        self.start_selection_t = 0.0
        self.global_selection = 0.0

        # Evaluation Time Variables
        self.start_evaluation_t = 0.0
        self.global_evaluation = 0.0

    def startTimer(self):
        """ start timing. """
        self.global_start_t = time.time()

    # ************************************************************
    def startTimeDataGenerating(self):
        """ Start counting time for the checkpoint """
        self.start_generating_t = time.time()

    def stopTimeDataGenerating(self):
        """ Accumulates time for checkpoints """
        self.global_generating += time.time() - self.start_generating_t

    # ************************************************************
    def startTimeCheckpoint(self):
        """ Start counting time for the checkpoint """
        self.start_checkpoint_t = time.time()

    def stopTimeCheckpoint(self):
        """ Accumulates time for checkpoints """
        self.checkpoint_time += time.time() - self.start_checkpoint_t

    # ************************************************************
    def startTimeMatching(self):
        """ Tracks MatchSet Time """
        self.start_matching_t = time.time()

    def stopTimeMatching(self):
        """ Tracks MatchSet Time """
        diff = time.time() - self.start_matching_t
        self.global_matching += diff

    # ************************************************************
    def startTimeDeletion(self):
        """ Tracks Deletion Time """
        self.start_deletion_t = time.time()

    def stopTimeDeletion(self):
        """ Tracks Deletion Time """
        diff = time.time() - self.start_deletion_t
        self.global_deletion += diff

    # ************************************************************
    def startTimeSubsumption(self):
        """Tracks Subsumption Time """
        self.start_subsumption_t = time.time()

    def stopTimeSubsumption(self):
        """Tracks Subsumption Time """
        diff = time.time() - self.start_subsumption_t
        self.global_subsumption += diff

    # ************************************************************
    def startTimeSelection(self):
        """ Tracks Selection Time """
        self.start_selection_t = time.time()

    def stopTimeSelection(self):
        """ Tracks Selection Time """
        diff = time.time() - self.start_selection_t
        self.global_selection += diff

    # ************************************************************
    def startTimeEvaluation(self):
        """ Tracks Evaluation Time """
        self.start_evaluation_t = time.time()

    def stopTimeEvaluation(self):
        """ Tracks Evaluation Time """
        diff = time.time() - self.start_evaluation_t
        self.global_evaluation += diff

    # ************************************************************
    def returnGlobalTimer(self):
        """ Set the global end timer, call at very end of algorithm. """
        # Reports time in minutes, added_time is for population reboot.
        self.global_time = (
                                       time.time() - self.global_start_t - self.checkpoint_time - self.global_generating) + self.added_time
        return self.global_time / 60.0

    def setTimerRestart(self, remakeFile):
        """ Sets all time values to the those previously evolved in the loaded popFile.  """
        try:
            stat_file = open(remakeFile + "_PopStats.txt", 'r')  # opens each datafile to read.
        except Exception as inst:
            print(type(inst))
            print(inst.args)
            print(inst)
            print('cannot open', remakeFile + "_PopStats.txt")
            raise

        time_data = 18

        line_tmp = None
        for i in range(time_data):
            line_tmp = stat_file.readline()
        list_tmp = line_tmp.strip().split('\t')
        self.added_time = float(list_tmp[1]) * 60  # previous global time added with Reboot.

        line_tmp = stat_file.readline()
        list_tmp = line_tmp.strip().split('\t')
        self.global_matching = float(list_tmp[1]) * 60

        line_tmp = stat_file.readline()
        list_tmp = line_tmp.strip().split('\t')
        self.global_deletion = float(list_tmp[1]) * 60

        line_tmp = stat_file.readline()
        list_tmp = line_tmp.strip().split('\t')
        self.global_subsumption = float(list_tmp[1]) * 60

        line_tmp = stat_file.readline()
        list_tmp = line_tmp.strip().split('\t')
        self.global_selection = float(list_tmp[1]) * 60

        line_tmp = stat_file.readline()
        list_tmp = line_tmp.strip().split('\t')
        self.global_evaluation = float(list_tmp[1]) * 60

        stat_file.close()

    ##############################################################################################

    def reportTimes(self):
        """ Reports the time summaries for this run. Returns a string ready to be printed out."""
        time_data = "Global Time\t" + str(self.global_time / 60.0) + \
                    "\nGenerating Time\t" + str(self.global_generating / 60.0) + \
                    "\nMatching Time\t" + str(self.global_matching / 60.0) + \
                    "\nDeletion Time\t" + str(self.global_deletion / 60.0) + \
                    "\nSubsumption Time\t" + str(self.global_subsumption / 60.0) + \
                    "\nSelection Time\t" + str(self.global_selection / 60.0) + \
                    "\nEvaluation Time\t" + str(self.global_evaluation / 60.0) + "\n"

        return time_data
