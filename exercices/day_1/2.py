#import lxmls.readers.simple_data_set as sds
import lxmls.readers.sentiment_reader as srs
import lxmls.classifiers.perceptron as percc
import matplotlib
import matplotlib.pyplot as plt

#sd = sds.SimpleDataSet(nr_examples=100, g1=[[-1, -1], 1], g2=[[1, 1], 1], balance=.5, split=[.5, 0, .5])
sd = srs.SentimentCorpus('books')
perc = percc.Perceptron()
params_perc_sd, params_per_epoch = perc.train(sd.train_X, sd.train_y)
y_pred_train = perc.test(sd.train_X, params_perc_sd)
acc_train = perc.evaluate(sd.train_y, y_pred_train)
y_pred_test = perc.test(sd.test_X, params_perc_sd)
acc_test = perc.evaluate(sd.test_y, y_pred_test)
#print(f'Perceptron Simple Dataset Accuracy train: {acc_train} test: {acc_test}')
print(f'Perceptron Amazon Sentiment Accuracy train: {acc_train} test: {acc_test}')

'''
# simple_data_set
fig, axis = sd.plot_data()
cm = matplotlib.colors.ListedColormap(['red', 'green', 'blue'])
for i in range(0, len(params_per_epoch), 5):
	fig, axis = sd.add_line(fig, axis, params_per_epoch[i], f'Perceptron {i}', cm(i))

plt.show()
'''
