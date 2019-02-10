class ClassAccuracy:
    def __init__(self):
        """ Initialize the accuracy calculation for a single class """
        self.T_myClass = 0  # For binary class problems this would include true positives
        self.T_otherClass = 0  # For binary class problems this would include true negatives
        self.F_myClass = 0  # For binary class problems this would include false positives
        self.F_otherClass = 0  # For binary class problems this would include false negatives

    def updateAccuracy(self, my_class, is_correct):
        """ Increment the appropriate cell of the confusion matrix """
        if my_class and is_correct:
            self.T_myClass += 1
        elif is_correct:
            self.T_otherClass += 1
        elif my_class:
            self.F_myClass += 1
        else:
            self.F_otherClass += 1
