import os
import gc
import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.classifiers import Classifier
import weka.core.serialization as serialization
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# Starting the JVM
jvm.start()

#Random Forest: Default
def random_forest(train_data):
	cls = Classifier(classname="weka.classifiers.trees.RandomForest")
	cls.build_classifier(train_data)
	return cls
	#pass
#
#Logistic Regression: Default
def logistic(train_data):
	cls = Classifier(classname="weka.classifiers.functions.Logistic")
	cls.build_classifier(train_data)
	return cls
	#pass
#
#Adaboost: Default
def adaboost(train_data):
	cls = Classifier(classname="weka.classifiers.meta.AdaBoostM1")
	cls.build_classifier(train_data)
	return cls
	#pass
#
#Bayes Network: Default
def bayes_net(train_data):
	cls = Classifier(classname="weka.classifiers.bayes.BayesNet")
	cls.build_classifier(train_data)
	return cls
	#pass
#
# Multi-Layer Perceptron Classifier: 2 hidden nodes in one hidden layer
def mlpc_2(train_data):
	cls = Classifier(classname="weka.classifiers.functions.MLPClassifier", options=["-N", "2"])
	cls.build_classifier(train_data)
	return cls
	#pass
#
# Naive Bayes
def naive_bayes(train_data):
	cls = Classifier(classname="weka.classifiers.bayes.NaiveBayes")
	cls.build_classifier(train_data)
	return cls
	#pass
#

# Classifiers
"""
algo_func_dict = {
'MLP_Classifier_2': mlpc_2,
'Bayes_Network': bayes_net,
'Random_Forest': random_forest,
'Adaboost': adaboost,
'Logistic_Regression': logistic,
}
"""
algo_func_dict = {
'Naive_Bayes': naive_bayes,
'Bayes_Network': bayes_net
}


# Saves the model in results_dir with the name of the algo: algo.model; mlpc2.model
def save_all_models(results_dir_str, train_data):
	c = 0
	for algo in algo_func_dict.keys():
		gc.collect()
		print "Training: " + str(algo)
		model = algo_func_dict[algo](train_data)
		out_file = results_dir_str + '/' + algo + ".model"
		serialization.write(out_file, model)
		c += 1
		print str(c) + ": Model Saved =>" + str(out_file)
	#
#

#Retrieves the classifier scores (Probability distribution per each class) from the given trained classifier model; columns(actual, predicted, proba_0, proba_1)
def get_classifier_score(trained_model, input_data, classes_list):
	num_of_classes = len(classes_list)
	header_list = ['inst_index', 'actual_class', 'predicted_class']
	for class_label in classes_list:
		header_list.append('proba_' + class_label)
	#
	header = np.array(header_list)
	for index, inst in enumerate(input_data):
		actual = inst.values[inst.class_index]
		pred = trained_model.classify_instance(inst)
		dist = trained_model.distribution_for_instance(inst)
		"""
		if len(dist) != num_of_classes:
			print "Error: Number of predited probabilities not equal to number of classes"
			return 1
		#"""
		#out_score = [inst_index, actual_class, predicted_class, proba_1, proba_2, ...]; inst_index starts from 1
		inst_index = index + 1
		inst_out_score = np.hstack((inst_index, actual, pred, dist))
		inst_out_score = [str(i) for i in inst_out_score]
		inst_out_score = ",".join(inst_out_score)
		inst_out_score = inst_out_score.split(',')
		inst_out_score = np.array(inst_out_score)
		if index == 0: #for the first instance, initializing total score. From the next iteration, instance scores are appended (vertically stacked to total score)
			total_out_score = inst_out_score
			#print inst_out_score
		else:
			total_out_score = np.vstack((total_out_score, inst_out_score))
	#
	#print total_out_score[:5] 
	final_scores_matrix = np.vstack((header, total_out_score))
	return final_scores_matrix
#

# Saves the scores in the results_dir from each of the model as algo_scores.csv; mlpc2_scores.csv
def get_scores_for_test_set(results_dir_str, model_file_list, test_data, classes_list):
	num_of_classes = len(classes_list)
	c = 0
	for model_file in model_file_list:
		gc.collect()
		print "Calculating Scores: " + model_file.split('/')[-1][:-6].upper()
		j_obj = serialization.read(model_file)
		trained_model = Classifier(jobject=j_obj)
		scores_matrix = get_classifier_score(trained_model, test_data, classes_list)
		out_file = model_file[:-6] + "_scores.csv"
		np.savetxt(out_file, scores_matrix, delimiter=",", fmt="%s")
		c += 1
		print str(c) + ": Test Scores Saved =>" + str(out_file)
#

#Plot a ROC curve for single algorithm
def plot_roc(class_name, targets, scores, ph):#ph - plot handler
	fpr, tpr, thresholds = roc_curve(targets, scores)
	roc_auc = auc(fpr, tpr)
	ph.plot(fpr, tpr, label=str(class_name) + '(%0.2f)' % roc_auc)
	return roc_auc
#
#Plot a PRC curve for single algorithm 
def plot_prc(class_name, targets, scores, ph):#ph - plot handler
	precision, recall, thresholds = precision_recall_curve(targets, scores)
	prc_auc = auc(recall, precision)
	ph.plot(recall, precision, label=str(class_name) + '(%0.2f)' % prc_auc)
	return prc_auc
#

# For each model-scores pair, evaluates certain metrics (Accuracy, RMSE, Area_under_ROC, Area_under_PRC) and writes the plots and summary in the results_dir
def evaluate_results(results_dir_str, model_scores_file_list, classes_list):
	num_of_classes = len(classes_list)
	summary_fp = open(results_dir_str + '/results.csv', 'w')
	summary_header = 'Algorithm,Accuracy,F1_Score,ROC_Area,PRC_Area' + '\n'
	summary_fp.write(summary_header)
	for model_scores_file in model_scores_file_list:
		print "Evaluating Metrics: " + model_scores_file.split('/')[-1][:-4]
		df = pd.read_csv(model_scores_file, sep=',',header=0)
		actual_target_df = df['actual_class']
		targets = actual_target_df.values
		predicted_df = df['predicted_class']
		preds = predicted_df.values
		# representing targets as binary labels in the form [n_samples, n_classes]
		# NOTE: Scores calculation would have converted it into classes_list to range(num_of_classes)
		classe_index_list = range(num_of_classes)
		targets_binary_labels = label_binarize(targets, classes=classe_index_list)
		scores_binary_labels = df.iloc[:, 3:].as_matrix()

		cur_algorithm = model_scores_file.split('/')[-1][:-11].upper()
		cur_accuracy = accuracy_score(targets, preds)
		cur_f1 = f1_score(targets, preds, average='weighted')

		cur_roc_auc = roc_auc_score(targets_binary_labels, scores_binary_labels, average ='weighted')
		cur_prc_auc = average_precision_score(targets_binary_labels, scores_binary_labels, average='weighted')  
		cur_results = [cur_algorithm, cur_accuracy, cur_f1, cur_roc_auc, cur_prc_auc]
		cur_results = [str(i) for i in cur_results]
		cur_results = ','.join(cur_results)
		summary_fp.write(cur_results + '\n')

		# Plotting ROC curves
		pdf_fh = PdfPages(model_scores_file[:-11] + '_roc.pdf')
		plt.clf()
		for class_index in classe_index_list:
			class_label = classes_list[class_index]
			cur_class_score_df = df['proba_' + class_label]
			cur_class_scores = cur_class_score_df.values
			cur_class_binary_targets = targets_binary_labels[:,class_index]
			plot_roc(class_label, cur_class_binary_targets, cur_class_scores, plt)
		#
		plt.plot([0, 1], [0, 1], 'k--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.0])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('ROC Curves: ' + cur_algorithm)
		plt.legend(loc="lower right", prop={'size':10})
		#plt.show()
		plt.savefig(pdf_fh, format='pdf')
		pdf_fh.close()
		
		# Plotting PRC curves
		pdf_fh = PdfPages(model_scores_file[:-11] + '_prc.pdf')
		plt.clf()
		for class_index in classe_index_list:
			class_label = classes_list[class_index]
			cur_class_score_df = df['proba_' + class_label]
			cur_class_scores = cur_class_score_df.values
			cur_class_binary_targets = targets_binary_labels[:,class_index]
			plot_prc(class_label, cur_class_binary_targets, cur_class_scores, plt)
		#
		plt.plot([1, 0], [0, 1], 'k--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.0])
		plt.xlabel('Recall')
		plt.ylabel('Precision')
		plt.title('Precision-Recall Curves: ' + cur_algorithm)
		plt.legend(loc="lower right", prop={'size':10})
		#plt.show()
		plt.savefig(pdf_fh, format='pdf')
		pdf_fh.close()
	#
	summary_fp.close()
#

# Run for one set of data: Results will be stored in ../results/label/
def run_analysis(label, train_data, test_data, classes_list):
	print 'Analysis running on: ' + label
	num_of_classes = len(classes_list)
	results_dir = '../results/' + label
	if not os.path.exists(results_dir):
		os.makedirs(results_dir)
	#
	#save_all_models(results_dir, train_data)
	gc.collect()

	model_file_list = []
	for algo in algo_func_dict.keys():
		model_file_list.append(results_dir + '/' + algo + '.model')
	#

	#get_scores_for_test_set(results_dir, model_file_list, test_data, classes_list)
	gc.collect()
	#

	model_scores_file_list = []
	for algo in algo_func_dict.keys():
		model_scores_file_list.append(results_dir + '/' + algo + '_scores.csv')
	#

	evaluate_results(results_dir, model_scores_file_list, classes_list)
#

# Web Services project calling
def run_webservices_project():
	loader = Loader(classname="weka.core.converters.ArffLoader")
	rt_train_data = loader.load_file('../data/rt_train_data.arff')
	rt_test_data = loader.load_file('../data/rt_test_data.arff')
	tp_train_data = loader.load_file('../data/tp_train_data.arff')
	tp_test_data = loader.load_file('../data/tp_test_data.arff')

	rt_train_data.class_is_last()
	rt_test_data.class_is_last()
	tp_train_data.class_is_last()
	tp_test_data.class_is_last()

	run_analysis('RT', rt_train_data, rt_test_data, ['RT0', 'RT1', 'RT2', 'RT3', 'RT4'])
	run_analysis('TP', tp_train_data, tp_test_data, ['TP0', 'TP1', 'TP2', 'TP3', 'TP4'])
#

run_webservices_project()
# Stopping the JVM
jvm.stop()
###

"""
label = 'RT'
classes_list = ['RT0', 'RT1', 'RT2', 'RT3', 'RT4']
results_dir = '../results/' + label
results_dir_str = results_dir
for algo in algo_func_dict.keys():
	model_scores_file_list.append(results_dir + '/' + algo + '_scores.csv')
#
model_scores_file = model_scores_file_list[0]
"""
