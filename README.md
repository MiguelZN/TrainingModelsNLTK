# NLP Homework 3 README
4/30/20
CISC489-NLP Topics
Shanker
**Group**: Muhammet Aydin, Max Luu, Brad Altmiller, Miguel Zavala

**Question 1 Instructions:**
&nbsp;
Notes:
- All question 1 code is contained in **Main.py**
- The three trained models were trained for 100 iterations and then saved using pickle as: 
**currpos_classifier100Iterations.pickle**, **currword_currpos_prevpos_classifier100Iterations.pickle**,
**currword_currpos_prevpos_nextword_nextpos_classifier100Iterations.pickle**
- Running Main.py will load all three existing pickle trained models and parse each sentence of 'HW3_test.txt' and print the baseNPs predicted by each model
- The accuracy of each training model was saved in the following text files:
**currpos_classifier100Iterations_OUTPUT.txt**,
**currword_currpos_prevpos_classifier100Iterations_OUTPUT.txt**,
**currword_currpos_prevpos_nextword_nextpos_classifier100Iterations_OUTPUT.txt**
- The printed training model Predicted BaseNPs output were already saved in the following text files:
**currpos_classifier100Iterations_PredictedBaseNPs.txt**,
**currword_currpos_prevpos_classifier100Iterations_PredictedBaseNPs.txt**,
**currword_currpos_prevpos_nextword_nextpos_classifier100Iterations_PredictedBaseNPs.txt**

&nbsp;
**To run our Question 1 training model code:**
```sh
$  python3 Main.py
```
&nbsp;
**To create a new training model:**
- set ConsecutiveNPChunker's parameter **InputtedClassifier** = **None**
- change ConsecutiveNPChunker's parameter **POSMODELTYPE =<POSMODELTYPES enumeration type>** 
(either: POSMODELTYPES.CURRPOS, POSMODELTYPES.CURRWORD_CURRPOS_PREVPOS, POSMODELTYPE.CURRWORD_CURRPOS_PREVPOS_NEXTWORD_NEXTPOS)
This represents which training features you would like to train on
- Set ConsecutiveNPChunker's parameter **iterations** = <int>
This represents the number of training iterations

```python
#EX: Creating a new training model using currpos for 5 training iterations 
newchunker = ConsecutiveNPChunker(train_sents, POSMODELTYPE=POSMODELTYPES.CURRPOS, iterations=5,InputtedClassifier=None)
getBaseNPsGivenChunkerAndTaggedSentences(newchunker, listOfPOSTaggedSentences)
```

After finished running:
- the model's accuracy and predicted baseNPs will be printed
- a new pickle file will be created with the name of the model 
EX: 'currpos_classifier1.pickle'
This newly created trained model can then be loaded for future use

**Loading existing trained models:**
- Set ConsecutiveNPChunker's parameter:
**InputtedClassifier = loadClassifier('classifierpicklefilepath')**
- EX: InputtedClassifier = loadClassifier('currpos_classifier1.pickle')

**Question 2 Instructions:**

**Question  Instructions:**
