import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
<<<<<<< HEAD
<<<<<<< HEAD
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
=======
>>>>>>> 6d4fb45... Base Code
=======
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
>>>>>>> ca20717... Submission_01
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> ca20717... Submission_01
    #raise NotImplementedError
    
    for testWord, (X,lengths) in test_set.get_all_Xlengths().items():
        prob = {}
        
        bestScore = None
        bestWord = None

        for trainedWord, model in models.items():
            try:
                score =  model.score(X,lengths)                
                
                if bestScore == None:
                    bestScore = score
                    bestWord = trainedWord
                else:
                    if score > bestScore:
                        bestScore = score
                        bestWord = trainedWord    
                prob[trainedWord] = score

            except:
                prob[trainedWord] = float("-inf")

        probabilities.append(prob)
        guesses.append(bestWord)
    return probabilities, guesses


<<<<<<< HEAD
=======
    raise NotImplementedError
>>>>>>> 6d4fb45... Base Code
=======
>>>>>>> ca20717... Submission_01
