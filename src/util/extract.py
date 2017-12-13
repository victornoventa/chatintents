#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, re, ast, itertools

file_absolute_path = sys.argv[1]
intents_list = ast.literal_eval(sys.argv[2])
figure_title = sys.argv[3]
figure_cmap = sys.argv[4]

times = []
scores = []
phrases_classifications = []
with open(file_absolute_path) as f:
    lines =  [x.strip('\n') for x in f.readlines()]
    for line in lines:
        if line[:5] == 'TIME:':
            times.append(line)
        elif line[:6] == 'SCORE:':
            scores.append(line)
        else:
            phrases_classifications.append(line)

predicted_classifications = []
correct_classifications = []
for classified_phrase in phrases_classifications:
    data = [x.split() for x in re.split(r"[\[\]]", classified_phrase)]
    if (len(data) > 1):
        predicted_classifications.append(data[1][1])
        correct_classifications.append(data[1][4])

from sklearn.metrics import confusion_matrix, classification_report
confusion_matrix_data = confusion_matrix(correct_classifications, predicted_classifications, labels=intents_list)
classification_report_data = classification_report(correct_classifications, predicted_classifications, labels=intents_list)

times_in_seconds = []
for time in times:
    data = time.split()
    times_in_seconds.append(data[1])

import numpy as np

times_in_seconds = np.array(times_in_seconds, dtype=np.float)

accepted_scores = []
total_scores = []
for score in scores:
    data = score.split()
    accepted_scores.append(data[1])
    total_scores.append(data[3])


accepted_scores = np.array(accepted_scores, dtype=np.float)
total_scores = np.array(total_scores, dtype=np.float)
score_means = accepted_scores / total_scores

file_absolute_path_without_extension = file_absolute_path.split('.')[0]
extracted_file = open(file_absolute_path_without_extension + '_extracted.txt', 'w')

extracted_file.write("EXTRACTED INFO")
extracted_file.write("\n\nconfusion_matrix\n")
extracted_file.write(np.array_str(confusion_matrix_data))

extracted_file.write("\n\nscore means\n")
extracted_file.write(np.array_str(score_means))
extracted_file.write("\naverage score\n")
extracted_file.write(np.array_str(np.average(score_means)))
extracted_file.write("\nscore std\n")
extracted_file.write(np.array_str(np.std(score_means)))

extracted_file.write("\n\nexecution times\n")
extracted_file.write(np.array_str(times_in_seconds))
extracted_file.write("\naverage time\n")
extracted_file.write(np.array_str(np.average(times_in_seconds)))
extracted_file.write("\ntime std\n")
extracted_file.write(np.array_str(np.std(times_in_seconds)))

extracted_file.write("\n\n" + classification_report_data)
extracted_file.close()

import matplotlib.pyplot as plt
# np.set_printoptions(precision=2)

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Plot non-normalized confusion matrix
fig = plt.figure(figsize=[12, 10])
plot_confusion_matrix(confusion_matrix_data, title=figure_title, classes=intents_list, cmap=figure_cmap)

fig.tight_layout()
plt.savefig(file_absolute_path_without_extension + '_confusion_matrix.png', format='png', dpi=150)
