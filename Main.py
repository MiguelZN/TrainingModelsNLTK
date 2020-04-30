'''
Miguel Zavala, Brad Altmiller, Max Luu ,Muhammet Aydin
CISC489-NLP Topics
Homework 3
'''

import nltk, pickle, os.path
from nltk.corpus import conll2000
from enum import Enum

# Uncomment to download more nltk datasets:
# nltk.download()
# nltk.download('averaged_perceptron_tagger')

'''
1)
Please  use  the  CoNLL  training  data  set 
(from  nltk.corpus,  import CoNLL2000  data) to train 3 different models. 
Different models are to be developed using the NLTK NP chunker (ConsecutiveNPChunker)  
(see  Section  3.3 from  the  above  link) by  using different sets of features as below:
a.Using only current pos
b.Using onlycurrent word, current pos and previous pos
c.Using only current word, current pos, previous pos and next word pos.

Compare the output on 10 sentences provided to you in HW3_test.txt. 
i.Manually annotate all baseNPs in these 10 sentences. Submit your annotation by listing the base NP strings for each sentence. 
ii.Next,for each of the trained  models,  find  their  predicted  baseNPs  and note  down  the  cases  where  they  differ  from  your  
annotation  and  where they are in agreement with your annotation.Note: For ConsecutiveNPChunker , please remove algorithm='megam'in the line 
self.classifier  =  nltk.MaxentClassifier.train(train_set,  algorithm='megam',  trace=0), so that the default algorithm is setto IIS.

'''
CLASSIFIER_SAVE_DIR = "" #Modify to preferred save location for classifier Object


# Enumeration type to organize the three model training types
class POSMODELTYPES(Enum):
    CURRPOS = "CURRPOS".lower()
    CURRWORD_CURRPOS_PREVPOS = "CURRWORD_CURRPOS_PREVPOS".lower()
    CURRWORD_CURRPOS_PREVPOS_NEXTWORD_NEXTPOS = "CURRWORD_CURRPOS_PREVPOS_NEXTWORD_NEXTPOS".lower()

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return self.value


# Takes in a list of sentences (strings) and assigns POS tags to each word (required for training)
def preprocess(listOfSentences):
    listOfPOSTaggedSentences = []
    for sentence in listOfSentences:
        tokenized_sentence = nltk.word_tokenize(sentence)
        pos_tagged_sentence = nltk.pos_tag(tokenized_sentence)
        print(pos_tagged_sentence)
        listOfPOSTaggedSentences.append(pos_tagged_sentence)

    return listOfPOSTaggedSentences


# Training Method: a.Using only current pos
def npchunk_featuresCurrPOS(sentence, i, history):
    word, pos = sentence[i]
    return {"pos": pos}


# Training Method: b.Using onlycurrent word, current pos and previous pos
def npchunk_featuresCurrWordCurrPOSPrevPOS(sentence, i, history):
    word, pos = sentence[i]
    prevword, prevpos = None, None

    if i == 0:
        prevword, prevpos = "<START>", "<START>"
    else:
        prevword, prevpos = sentence[i - 1]
    return {"pos": pos, "word": word, "prevpos": prevpos}


# Training Method: c.Using only current word, current pos, previous pos and next word pos
def npchunk_featuresCurrWordCurrPOSPrevPOSNextWordNextPOS(sentence, i, history):
    word, pos = sentence[i]
    preword, prevpos, nextword, nextpos = None, None, None, None

    if i == 0:
        prevword, prevpos = "<START>", "<START>"
    else:
        prevword, prevpos = sentence[i - 1]
    if i == len(sentence) - 1:
        nextword, nextpos = "<END>", "<END>"
    else:
        nextword, nextpos = sentence[i + 1]
    return {"pos": pos, "word": word, "prevpos": prevpos, "nextpos": nextpos, "nextword": nextword}


# NLTK method for training
class ConsecutiveNPChunkTagger(nltk.TaggerI):
    def __init__(self, train_sents, posmodel=POSMODELTYPES.CURRPOS, iterations=1, InputtedClassifier=None):
        self.POSMODELTYPE = posmodel  # Indicates which model to use
        train_set = []

        # Checks if an existing classifier Object was not inputted
        # If no classifier was given then recreates the classifier Object
        # (trains using the conll2000 dataset)
        if (InputtedClassifier == None):
            for tagged_sent in train_sents:
                untagged_sent = nltk.tag.untag(tagged_sent)
                history = []
                for i, (word, tag) in enumerate(tagged_sent):
                    featureset = None

                    # Only uses the current POS:
                    if (self.POSMODELTYPE == POSMODELTYPES.CURRPOS):
                        featureset = npchunk_featuresCurrPOS(untagged_sent, i, history)
                    elif (self.POSMODELTYPE == POSMODELTYPES.CURRWORD_CURRPOS_PREVPOS):
                        featureset = npchunk_featuresCurrWordCurrPOSPrevPOS(untagged_sent, i, history)
                    else:
                        featureset = npchunk_featuresCurrWordCurrPOSPrevPOSNextWordNextPOS(untagged_sent, i, history)
                    train_set.append((featureset, tag))
                    history.append(tag)

            print("FINISHED TRAIN_SET:")
            # print(train_set)
            self.classifier = nltk.MaxentClassifier.train(train_set, max_iter=iterations)  # , trace = 0

            # Saves the classifier using pickle (In order to save time):
            id = 1  # starts at id 1 to check if file exists
            classifierPickleFile = CLASSIFIER_SAVE_DIR+str(self.POSMODELTYPE.__str__()) + "_classifier" + str(id) + ".pickle" #File path for this classifer Object
            print(classifierPickleFile)
            while (True):
                if (os.path.isfile(classifierPickleFile)):
                    classifier_f = open(classifierPickleFile, "rb")
                    classifier = pickle.load(classifier_f)
                    print(classifier)
                    classifier_f.close()

                    id += 1
                    classifierPickleFile = CLASSIFIER_SAVE_DIR+str(self.POSMODELTYPE) + "_classifier" + str(
                        id) + ".pickle"  # Tries a new pickle file name
                else:
                    # Once it finds an available file path, it saves the classifier
                    saveClassifier = open(classifierPickleFile, "wb")
                    pickle.dump(self.classifier, saveClassifier)
                    print("PRINTING CLASSIFIER")
                    print(self.classifier)
                    saveClassifier.close()
                    break
            print("FINISHED CLASSIFIER")
        # If an existing classifier Object was given (existing training model)
        # Then simply uses it rather than recreating it
        else:
            print("Was given an already existing classifier! Loading classifier..")
            self.classifier = InputtedClassifier

    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = None
            # Only uses the current POS:
            if (self.POSMODELTYPE == POSMODELTYPES.CURRPOS):
                featureset = npchunk_featuresCurrPOS(sentence, i, history)
            elif (self.POSMODELTYPE == POSMODELTYPES.CURRWORD_CURRPOS_PREVPOS):
                featureset = npchunk_featuresCurrWordCurrPOSPrevPOS(sentence, i, history)
            else:
                featureset = npchunk_featuresCurrWordCurrPOSPrevPOSNextWordNextPOS(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)


class ConsecutiveNPChunker(nltk.ChunkParserI):
    def __init__(self, train_sents, POSMODELTYPE=POSMODELTYPES.CURRPOS, iterations: int = 1, InputtedClassifier=None):
        tagged_sents = [[((w, t), c) for (w, t, c) in
                         nltk.chunk.tree2conlltags(sent)]
                        for sent in train_sents]

        print("USING METHOD:" + str(POSMODELTYPE))
        self.tagger = ConsecutiveNPChunkTagger(tagged_sents, POSMODELTYPE, iterations, InputtedClassifier)

    def parse(self, sentence):
        tagged_sents = self.tagger.tag(sentence)
        conlltags = [(w, t, c) for ((w, t), c) in tagged_sents]
        return nltk.chunk.conlltags2tree(conlltags)


# Loads an existing classifier Object by inputting its path
# Must be a python '.pickle' file
def loadClassifier(classifierpath: str):
    classifier_f = open(classifierpath, "rb")
    classifier = pickle.load(classifier_f)
    classifier_f.close()
    return classifier


# Returns a list of strings representing line by line in a text file
def getLineByLine(textfile):
    file = open(textfile, 'r')
    listOfLines = []
    for line in file.readlines():
        # print(line)
        listOfLines.append(line.rstrip('\n\r'))

    file.close()
    return listOfLines


# Returns the Predicted BaseNPs for the given classifier Object, takes in a list of sentences(strs)
def getBaseNPsGivenChunkerAndTaggedSentences(chunker: nltk.ChunkParserI, taggedsentences: list):
    baseNPs = {}
    sentence_id = 1
    print("Printing Predicted BaseNPs:")
    for sentence in taggedsentences:
        baseNPs[sentence_id] = chunker.parse(sentence)
        print("Sentence " + str(sentence_id) + ":")
        print(baseNPs[sentence_id])

        sentence_id += 1

    return baseNPs


# Main------------------------------------------------------
print('main')

# Gets the training sentences and runs chunker on them to get classifier
train_sents = conll2000.chunked_sents('train.txt')
print("Starting Chunker..(wait a few mins)")

# Extracts line by line sentences of our 'HW3_test.txt'
hw3_sentences = getLineByLine('HW3_test.txt')

# Assigns POS Tags each of the words in each sentence
listOfPOSTaggedSentences = preprocess(hw3_sentences)

# Training Method: a.Using only current pos
chunkerUsingCurrPOS = ConsecutiveNPChunker(train_sents, POSMODELTYPE=POSMODELTYPES.CURRPOS, iterations=1,InputtedClassifier=None)
getBaseNPsGivenChunkerAndTaggedSentences(chunkerUsingCurrPOS, listOfPOSTaggedSentences)

print("-----------------------")

# Training Method: b.Using only current word, current pos and previous pos
#chunkerUsingCurrWordCurrPOSPrevPOS = ConsecutiveNPChunker(train_sents,POSMODELTYPE=POSMODELTYPES.CURRWORD_CURRPOS_PREVPOS,iterations=100, InputtedClassifier=loadClassifier('currword_currpos_prevpos_classifier100Iterations.pickle'))
#getBaseNPsGivenChunkerAndTaggedSentences(chunkerUsingCurrWordCurrPOSPrevPOS, listOfPOSTaggedSentences)

print("-----------------------")
# Training Method: c.Using only current word, current pos, previous pos and next word pos
#chunkerUsingCurrWordCurrPOSPrevPOSNextWordNextPOS = ConsecutiveNPChunker(train_sents,POSMODELTYPE=POSMODELTYPES.CURRWORD_CURRPOS_PREVPOS_NEXTWORD_NEXTPOS,iterations=100,InputtedClassifier=loadClassifier('currword_currpos_prevpos_nextword_nextpos_classifier100Iterations.pickle'))
#getBaseNPsGivenChunkerAndTaggedSentences(chunkerUsingCurrWordCurrPOSPrevPOSNextWordNextPOS, listOfPOSTaggedSentences)
