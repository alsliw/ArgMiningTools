# ArgMiningTools

This project contains up to 8 different Argument Mining Tools capable in predicting a sentence in English language as either argumentative or non-argumentative.

## Installation of package dependencies

- pip install -r requirements.txt

## Software dependencies

- Python 2.7

## Data Insertion

The input requires text files located in *src/data/original*. <br />
The application automatically generates a corresponding .csv file with output results for each input text file stored in *src/data/updated*. <br />
Note that for input documents each line is assumed as one sentence.

## Run Application

Navigate to the directory *src* and run python Processor.py

## Models Selection

There is a possibility to select N out of 8 models in [property.txt](src/property.txt). <br />
Each sentence of an input document is predicted as majority vote of the N selected models. <br />
Models with a "1" in usage field indicate as being selected and a "0" as non-selected. <br />
The property file contains following models:
- Linear Discriminant Analysis (LDA)
- C-Support Vector Classification (SVM)
- Logistic Regression (LR)
- Random Forest (RF)
- AdaBoost classifier (ADA)
- K Nearest Neighbor (KNN)
- Gaussian Naive Bayes (GNB)
- Long short-term memory (LSTM)

In terms of LSTM we adapted the Neural Architectures for Named Entity Recognition described in Lample et al, 2016.

## Word Embeddings

We use a pre-trained word embeddings from Google News corpus in order to use this for LSTM application. <br />
- Word2Vec: https://code.google.com/p/word2vec/	

Insert the binary file in *src* package

## Authors

* **Alfred Sliwa**  (alfred.sliwa.92@stud.uni-due.de)
* **Ahmet Aker** (aker@is.inf.uni-due.de)

## License

This project is licensed under the GNU LGPL - see the [LICENSE.md](LICENSE.md) file for details

## Citation

If you use ArgMiningTools in your research, please cite the paper: <br />
```
@inproceedings{aker2017works,
	title={What works and what does not: Classifier and feature analysis for argument mining},
	author={Aker, Ahmet and Sliwa, Alfred and Ma, Yuan and Lui, Ruishen and Borad, Niravkumar and Ziyaei, Seyedeh
	and Ghobadi, Mina},
	booktitle={Proceedings of the 4th Workshop on Argument Mining},
	pages={91--96},
	year={2017}
}
```
