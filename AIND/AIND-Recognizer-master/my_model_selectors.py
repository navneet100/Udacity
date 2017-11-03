import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
<<<<<<< HEAD
<<<<<<< HEAD
        #print(all_word_sequences)
=======
>>>>>>> 6d4fb45... Base Code
=======
        #print(all_word_sequences)
>>>>>>> ca20717... Submission_01
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
<<<<<<< HEAD
<<<<<<< HEAD
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        
=======
>>>>>>> 6d4fb45... Base Code
=======
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        
>>>>>>> ca20717... Submission_01
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
<<<<<<< HEAD
<<<<<<< HEAD
    """ select the model with the lowest Baysian Information Criterion(BIC) score
=======
    """ select the model with the lowest Bayesian Information Criterion(BIC) score
>>>>>>> 6d4fb45... Base Code
=======
    """ select the model with the lowest Baysian Information Criterion(BIC) score
>>>>>>> ca20717... Submission_01

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> ca20717... Submission_01
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            minBIC = None
            bestModel = None
            
            num_states = self.min_n_components            
           
            while  num_states <= self.max_n_components:
                
                model = self.base_model(num_states) 
                
                logL  = model.score(self.X, self.lengths )
                
                #initial probabilities + transition probabilites + Emission probablities
                #num_Features = len(self.X[0])
                num_Features = model.n_features
                #p =  (num_states - 1)   + (num_states * ( num_states - 1)) + (2 * num_states * num_Features)
                #simplification of above
                p =  num_states * num_states - 1   + (2 * num_states * num_Features)
                
                num_data_points = len(self.X)
                
                BIC = -2.0 * logL + p * np.log(num_data_points)
                
                if minBIC == None:
                    minBIC = BIC
                    bestModel = model
                else:
                    if BIC < minBIC:
                        minBIC = BIC
                        bestModel = model
                num_states += 1           
        except:
            return self.base_model(self.n_constant)
        
        return bestModel 
<<<<<<< HEAD

            
=======

        # TODO implement model selection based on BIC scores
        raise NotImplementedError


>>>>>>> 6d4fb45... Base Code
=======

            
>>>>>>> ca20717... Submission_01
class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
<<<<<<< HEAD
<<<<<<< HEAD
=======
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
>>>>>>> 6d4fb45... Base Code
=======
>>>>>>> ca20717... Submission_01
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
<<<<<<< HEAD
<<<<<<< HEAD
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # TODO implement model selection based on DIC scores
        #raise NotImplementedError
        try:
            bestDIC = None
            bestModel = None
            
            for n in range(self.min_n_components, self.max_n_components + 1):        
                model = self.base_model(n)            
                logLCurrent  = model.score(self.X, self.lengths )
                
                otherWordScores = []
                
                for word, (X, lengths) in self.hwords.items():
                    if word != self.this_word:        
                        otherWordScores.append(model.score(X, lengths))
                        
                meanOthersLogL = np.mean(otherWordScores)
                
                diffLogL = logLCurrent - meanOthersLogL
                if bestDIC == None:
                    bestDIC = diffLogL
                    bestModel = model
                else:
                    if diffLogL > bestDIC:
                        bestDIC = diffLogL
                        bestModel = model
       
        except:
            return self.base_model(self.n_constant)
        return bestModel 
=======

        # TODO implement model selection based on DIC scores
        raise NotImplementedError

>>>>>>> 6d4fb45... Base Code
=======
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # TODO implement model selection based on DIC scores
        #raise NotImplementedError
        try:
            bestDIC = None
            bestModel = None
            
            for n in range(self.min_n_components, self.max_n_components + 1):        
                model = self.base_model(n)            
                logLCurrent  = model.score(self.X, self.lengths )
                
                otherWordScores = []
                
                for word, (X, lengths) in self.hwords.items():
                    if word != self.this_word:        
                        otherWordScores.append(model.score(X, lengths))
                        
                meanOthersLogL = np.mean(otherWordScores)
                
                diffLogL = logLCurrent - meanOthersLogL
                if bestDIC == None:
                    bestDIC = diffLogL
                    bestModel = model
                else:
                    if diffLogL > bestDIC:
                        bestDIC = diffLogL
                        bestModel = model
       
        except:
            return self.base_model(self.n_constant)
        return bestModel 
>>>>>>> ca20717... Submission_01

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
<<<<<<< HEAD
<<<<<<< HEAD
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        # TODO implement model selection using CV
        #raise NotImplementedError
        num_folds = 2
        
        try:
            kf = KFold(n_splits = num_folds)
            
            bestCV = None
            bestModel = None
            
            scoreCV = []
            
            for n in range(self.min_n_components, self.max_n_components + 1):        
                for trainIndices, testIndices in kf.split(self.sequences):  
                    self.X, self.lengths = combine_sequences(trainIndices, self.sequences)   
                    model = self.base_model(n)                    
                    X, l = combine_sequences(testIndices, self.sequences)
                    scoreCV.append(model.score(X, l))   
                
                avgScore = np.mean(scoreCV)
                if bestCV == None:
                    bestCV = avgScore
                    bestModel = model
                else:
                    if avgScore > bestCV:
                        bestCV = avgScore
                        bestModel = model    
        except:
            return self.base_model(self.n_constant)
        
        return bestModel 
=======

        # TODO implement model selection using CV
        raise NotImplementedError
>>>>>>> 6d4fb45... Base Code
=======
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        # TODO implement model selection using CV
        #raise NotImplementedError
        num_folds = 2
        
        try:
            kf = KFold(n_splits = num_folds)
            
            bestCV = None
            bestModel = None
            
            scoreCV = []
            
            for n in range(self.min_n_components, self.max_n_components + 1):        
                for trainIndices, testIndices in kf.split(self.sequences):  
                    self.X, self.lengths = combine_sequences(trainIndices, self.sequences)   
                    model = self.base_model(n)                    
                    X, l = combine_sequences(testIndices, self.sequences)
                    scoreCV.append(model.score(X, l))   
                
                avgScore = np.mean(scoreCV)
                if bestCV == None:
                    bestCV = avgScore
                    bestModel = model
                else:
                    if avgScore > bestCV:
                        bestCV = avgScore
                        bestModel = model    
        except:
            return self.base_model(self.n_constant)
        
        return bestModel 
>>>>>>> ca20717... Submission_01
