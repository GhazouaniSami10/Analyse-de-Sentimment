from sklearn.feature_extraction.text import CountVectorizer #bibliothèque libre Python destinée à l'apprentissage automatique
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix

import numpy as np
import itertools
import matplotlib.pyplot as plt
import pandas 
#################################
def get_all_data():
    root = "Data/"

    with open(root + "imdb_labelled.txt", "r") as text_file:  #IMDb is the world's most popular and authoritative source for movie, TV and celebrity content. Find ratings and reviews for the newest movie and TV shows.
        data = text_file.read().split('\n')
         
    with open(root + "amazon_cells_labelled.txt", "r") as text_file:#site de vente en ligne 
        data += text_file.read().split('\n')

    with open(root + "yelp_labelled.txt", "r") as text_file:##User Reviews and Recommendations of Best Restaurants, Shopping, Nightlife, Food, Entertainment, Things to Do, Services and More at Yelp.
        data += text_file.read().split('\n')

    return data
#get_all_data()
values = get_all_data()
print(values[0])
print(values[10])
print(values[100])
print(values[1000])
##############################
def preprocessing_data(data):
    processing_data = []
    for single_data in data:
        if len(single_data.split("\t")) == 2 and single_data.split("\t")[1] != "":
            processing_data.append(single_data.split("\t"))

    return processing_data

all_data = get_all_data()
values = preprocessing_data(all_data)
print(values[0])
print(values[10])
#########################
def split_data(data):
    total = len(data)
    training_ratio = 0.75
    training_data = []
    evaluation_data = []

    for indice in range(0, total):
        if indice < total * training_ratio:
            training_data.append(data[indice])
        else:
            evaluation_data.append(data[indice])

    return training_data, evaluation_data
    ##############################
def preprocessing_step():
    data = get_all_data()
    processing_data = preprocessing_data(data)

    return split_data(processing_data)
def training_step(data, vectorizer):
    training_text = [data[0] for data in data]
    training_result = [data[1] for data in data]

    training_text = vectorizer.fit_transform(training_text)

    return BernoulliNB().fit(training_text, training_result)

training_data, evaluation_data = preprocessing_step()
vectorizer = CountVectorizer(binary = 'true')
classifier = training_step(training_data, vectorizer)
result = classifier.predict(vectorizer.transform(["I love this movie!"]))

result[0]
#################
def analyse_text(classifier, vectorizer, text):
    return text, classifier.predict(vectorizer.transform([text]))

new_result = analyse_text(classifier, vectorizer, "Best product ever")
new_result
##################
def print_result(result):
    text, analysis_result = result
    print_text = "Positive" if analysis_result[0] == '1' else "Negative"
    print(text, ":", print_text)
    
print_result(new_result)

#Best product ever : Positive
##########################
print_result( analyse_text(classifier, vectorizer,"this is the best movie"))
print_result( analyse_text(classifier, vectorizer,"this is the worst movie"))
print_result( analyse_text(classifier, vectorizer,"awesome!"))
print_result( analyse_text(classifier, vectorizer,"10/10"))
print_result( analyse_text(classifier, vectorizer,"so bad"))
print_result( analyse_text(classifier, vectorizer,"nice"))
print_result( analyse_text(classifier, vectorizer,"very very nice"))
print_result( analyse_text(classifier, vectorizer,"fack you"))
print_result( analyse_text(classifier, vectorizer,"you are very bad"))
#######################
def simple_evaluation(evaluation_data):
    evaluation_text     = [evaluation_data[0] for evaluation_data in evaluation_data]
    evaluation_result   = [evaluation_data[1] for evaluation_data in evaluation_data]

    total = len(evaluation_text)
    corrects = 0
    for index in range(0, total):
        analysis_result = analyse_text(classifier, vectorizer, evaluation_text[index])
        text, result = analysis_result
        corrects += 1 if result[0] == evaluation_result[index] else 0

    return corrects * 100 / total

simple_evaluation(evaluation_data)
#####################
def create_confusion_matrix(evaluation_data):
    evaluation_text     = [evaluation_data[0] for evaluation_data in evaluation_data]
    actual_result       = [evaluation_data[1] for evaluation_data in evaluation_data]
    prediction_result   = []
    for text in evaluation_text:
        analysis_result = analyse_text(classifier, vectorizer, text)
        prediction_result.append(analysis_result[1][0])
    
    matrix = confusion_matrix(actual_result, prediction_result)
    return matrix
    
confusion_matrix_result = create_confusion_matrix(evaluation_data)
##########################
import pandas as pd
pandas.DataFrame(confusion_matrix_result, columns=["Negatives", "Positives"],index=["Negatives", "Positives"])
#################
classes = ["Negatives", "Positives"]

plt.figure()
plt.imshow(confusion_matrix_result, interpolation='nearest', cmap=plt.cm.Greens)
plt.title("Confusion Matrix - Sentiment Analysis",size=20)
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

text_format = 'd'
thresh = confusion_matrix_result.max() / 2.
for row, column in itertools.product(range(confusion_matrix_result.shape[0]), range(confusion_matrix_result.shape[1])):
    plt.text(column, row, format(confusion_matrix_result[row, column], text_format),
             horizontalalignment="center",
             color="white" if confusion_matrix_result[row, column] > thresh else "black")

plt.ylabel('True label',size = 30, color='r')
plt.xlabel('Predicted label',size = 30, color='b')
plt.tight_layout()
#plt.grid()
plt.show()
#####################    