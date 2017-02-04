"""
Hidden Markov Model sequence tagger
"""
from classifier import Classifier
import numpy as np
from collections import defaultdict as dd
from collections import Counter

class HMM(Classifier):
    def get_model(self): return self.seen
    def set_model(self, model): self.seen = model
    model = property(get_model, set_model)

    def __init__(self):
        super(HMM, self).__init__(self)
        self.model = {}
        self.UNK = '<UNK>'

    def _collect_counts(self, instance_list):
        """Collect counts necessary for fitting parameters

        This function updates self.model['transition_matrix'] and self.model['emission_matrix']
        based on the counts for sequence from this new given instance

        Returns None
        """
        for instance in instance_list:
            for label, feature in zip(instance.label, instance.feature_vector):
                i = self.model['label2index'][label]
                j = self.model['feature2index'][self.UNK]
                if feature in self.model['features']:
                    j = self.model['feature2index'][feature]
                self.model['emission_matrix'][i, j] += 1.0
            for i, label in enumerate(instance.label):
                if i == len(instance.label) - 1:
                    self.model['final_probabilities'][label] += 1.0
                if i == 0:
                    self.model['initial_probabilities'][label] += 1.0
                else:
                    prev = instance.label[i-1]
                    i = self.model['label2index'][prev]
                    j = self.model['label2index'][label]
                    self.model['transition_matrix'][i, j] += 1.0

    def train(self, instance_list):
        """Fit parameters for hidden markov model

        Takes in training data and constructs the model for the labels and features.
        Smoothing is +1 smoothing by initializing np.ones().

        Model is stored as a dictionary of {'labels': [set of labels], 'features': [set of all
        features that appeared more than 3 times], 'label2index': {dictionary of label:index pairs},
        'index2label' {dictionary of index:label pairs}, 'transition_matrix': <numpy ndarry>,
        'emission_matrix': <numpy ndarry>, 'initial_probabilities': {dictionary of label:prob pairs},
        'final_probabilities': {dictionary of label:prob pairs}}

        The probibility dictionaries and matrices are stored as counts from _collect_counts() and
        then normalized to probabilities to be used for decoding.

        Returns None
        """
        labels = set()
        features = Counter()
        for instance in instance_list:
            instance.features()
            labels |= set(instance.label)
            features.update(instance.feature_vector)
        features = set(feature for feature, count in features.items() if count > 3)
        features.add(self.UNK)
        self.model['labels'] = labels
        self.model['features'] = features
        self.model['label2index'] = {}
        self.model['index2label'] = {}
        self.model['feature2index'] = {}
        self.model['index2feature'] = {}
        for i, label in enumerate(labels):
            self.model['label2index'][label] = i
            self.model['index2label'][i] = label
        for i, feature in enumerate(features):
            self.model['feature2index'][feature] = i
            self.model['index2feature'][i] = feature

        self.model['transition_matrix'] = np.ones((len(labels), len(labels)))
        self.model['emission_matrix'] = np.ones((len(labels), len(features)))
        self.model['initial_probabilities'] = dd(int)
        self.model['final_probabilities'] = dd(int)

        self._collect_counts(instance_list)

        # normalize initial and final transition probabilities
        denom = sum(self.model['initial_probabilities'].values())
        for key in self.model['initial_probabilities']:
            self.model['initial_probabilities'][key] /= denom
        denom = sum(self.model['final_probabilities'].values())
        for key in self.model['final_probabilities']:
            self.model['final_probabilities'][key] /= denom

        # normalize transition matrix and emission matrix probabilities
        self.model['transition_matrix'] /= self.model['transition_matrix'].sum(axis=0)
        self.model['emission_matrix'] /= self.model['emission_matrix'].sum(axis=0)

    def classify(self, instance):
        """Viterbi decoding algorithm

        Runs dynamic_programming_on_trellis in Viterbi mode to get back_pointers. Then does the
        finalization step and runs back through back pointers to get the the sequence and returns it.

        Returns a list of labels - e.g. ['B','I','O','O','B']
        """
        instance.features()
        trellis, backtrace_pointers = self.dynamic_programming_on_trellis(instance, False)
        values = np.zeros(len(self.model['labels']))
        for label in self.model['labels']:
            i = self.model['label2index'][label]
            values[i] = trellis[i, -1] * self.model['final_probabilities'][label]

        best_sequence = [np.argmax(values)]
        for t in range(len(instance.feature_vector)-1, 0, -1):
            best_sequence.append(backtrace_pointers[int(best_sequence[-1]), t])
        best_sequence = [self.model['index2label'][e] for e in best_sequence[::-1]]
        return best_sequence

    def compute_observation_loglikelihood(self, instance):
        """Compute and return log P(X|parameters) = loglikelihood of observations"""
        trellis = self.dynamic_programming_on_trellis(instance, True)
        loglikelihood = 0.0
        return loglikelihood

    def dynamic_programming_on_trellis(self, instance, run_forward_alg=True):
        """Run Forward algorithm or Viterbi algorithm

        This function uses the trellis to implement dynamic
        programming algorithm for obtaining the best sequence
        of labels given the observations

        Returns trellis filled up with the forward probabilities and backtrace pointers
        for finding the best sequence in Viterbi mode, and only the trellis in Forward mode.
        """
        trellis = np.zeros((len(self.model['labels']), len(instance.feature_vector)))
        backtrace_pointers = trellis.copy()
        for label in self.model['labels']:
            i = self.model['label2index'][label]
            a = self.model['initial_probabilities'][label]
            j = self.model['feature2index'][self.UNK]
            if instance.feature_vector[0] in self.model['features']:
                j = self.model['feature2index'][instance.feature_vector[0]]
            b = self.model['emission_matrix'][i, j]
            trellis[i, 0] = a * b
        for t in range(1, len(instance.feature_vector)):
            for label in self.model['labels']:
                alpha = trellis[:, t - 1]
                i = self.model['label2index'][label]
                a = self.model['transition_matrix'][:, i]
                j = self.model['feature2index'][self.UNK]
                if instance.feature_vector[t] in self.model['features']:
                    j = self.model['feature2index'][instance.feature_vector[t]]
                b = self.model['emission_matrix'][i, j]
                if run_forward_alg:
                    trellis[i, t] = sum(alpha * a * b)
                else:
                    trellis[i, t] = max(alpha * a * b)
                    backtrace_pointers[i, t] = np.argmax(alpha * a)
        if run_forward_alg:
            return trellis
        return trellis, backtrace_pointers

    def train_semisupervised(self, unlabeled_instance_list, labeled_instance_list=None):
        """Baum-Welch algorithm for fitting HMM from unlabeled data (EXTRA CREDIT)

        The algorithm first initializes the model with the labeled data if given.
        The model is initialized randomly otherwise. Then it runs
        Baum-Welch algorithm to enhance the model with more data.

        Add your docstring here explaining how you implement this function

        Returns None
        """
        if labeled_instance_list is not None:
            self.train(labeled_instance_list)
        else:
            # TODO: initialize the model randomly
            pass
        while True:
            # E-Step
            self.expected_transition_counts = np.zeros((1, 1))
            self.expected_feature_counts = np.zeros((1, 1))
            for instance in instance_list:
                (alpha_table, beta_table) = self._run_forward_backward(instance)
                # TODO: update the expected count tables based on alphas and betas
                # also combine the expected count with the observed counts from the labeled data
                # M-Step
                # TODO: reestimate the parameters
            if self._has_converged(old_likelihood, likelihood):
                break

    def _has_converged(self, old_likelihood, likelihood):
        """Determine whether the parameters have converged or not (EXTRA CREDIT)

        Returns True if the parameters have converged.
        """
        return True

    def _run_forward_backward(self, instance):
        """Forward-backward algorithm for HMM using trellis (EXTRA CREDIT)

        Fill up the alpha and beta trellises (the same notation as
        presented in the lecture and Martin and Jurafsky)
        You can reuse your forward algorithm here

        return a tuple of tables consisting of alpha and beta tables
        """
        alpha_table = np.zeros((1, 1))
        beta_table = np.zeros((1, 1))
        # TODO: implement forward backward algorithm right here

        return (alpha_table, beta_table)
