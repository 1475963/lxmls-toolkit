import lxmls.readers.simple_data_set as sds
import lxmls.readers.sentiment_reader as srs
import lxmls.classifiers.mira as mirac
import matplotlib.pyplot as plt

sd = sds.SimpleDataSet(nr_examples=100, g1=[[-1, -1], 1], g2=[[1, 1], 1], balance=.5, split=[.5, 0, .5])
mira = mirac.Mira()
mira.regularizer = 1.0 # lambda
params_mira_sd = mira.train(sd.train_X, sd.train_y)
y_pred_train = mira.test(sd.train_X, params_mira_sd)
acc_train = mira.evaluate(sd.train_y, y_pred_train)
y_pred_test = mira.test(sd.test_X, params_mira_sd)
acc_test = mira.evaluate(sd.test_y, y_pred_test)
print(f'Mira Simple Dataset Accuracy train: {acc_train} test: {acc_test}')

fig, axis = sd.plot_data()
fig, axis = sd.add_line(fig, axis, params_mira_sd, 'Mira', 'green')
plt.gca()
plt.gcf()
plt.show()

scr = srs.SentimentCorpus('books')
params_mira_sd = mira.train(scr.train_X, scr.train_y)
y_pred_train = mira.test(scr.train_X, params_mira_sd)
acc_train = mira.evaluate(scr.train_y, y_pred_train)
y_pred_test = mira.test(scr.test_X, params_mira_sd)
acc_test = mira.evaluate(scr.test_y, y_pred_test)
print(f'Mira Amazon Sentiment Accuracy train: {acc_train} test: {acc_test}')
