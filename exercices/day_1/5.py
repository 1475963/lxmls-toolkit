import lxmls.readers.simple_data_set as sds
import lxmls.readers.sentiment_reader as srs
import lxmls.classifiers.svm as svmc
import matplotlib.pyplot as plt

sd = sds.SimpleDataSet(nr_examples=100, g1=[[-1, -1], 1], g2=[[1, 1], 1], balance=.5, split=[.5, 0, .5])
scr = srs.SentimentCorpus('books')
svm = svmc.SVM()
svm.regularizer = 0.6
params_svm_sd = svm.train(sd.train_X, sd.train_y)
y_pred_train = svm.test(sd.train_X, params_svm_sd)
acc_train = svm.evaluate(sd.train_y, y_pred_train)
y_pred_test = svm.test(sd.test_X, params_svm_sd)
acc_test = svm.evaluate(sd.test_y, y_pred_test)
print(f'SVM Online Simple Dataset Accuracy train: {acc_train} test: {acc_test}')

fig, axis = sd.plot_data()
fig, axis = sd.add_line(fig, axis, params_svm_sd, 'SVM', 'orange')

plt.show()

params_svm_scr = svm.train(scr.train_X, scr.train_y)
y_pred_train = svm.test(scr.train_X, params_svm_scr)
acc_train = svm.evaluate(scr.train_y, y_pred_train)
y_pred_test = svm.test(scr.test_X, params_svm_scr)
acc_test = svm.evaluate(scr.test_y, y_pred_test)
print(f'SVM Online Amazon Sentiment Accuracy train: {acc_train} test: {acc_test}')
