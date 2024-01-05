from utils_fingerprint import *
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score
from sklearn.metrics import classification_report
import os
from rdkit import Chem
import xlwt
import sklearn
from sklearn.metrics import roc_curve, auc

save_data_path = './data/SVM_Results/'  # 修改路径部分,保存路径
os.makedirs(save_data_path, exist_ok=True)

# load data
with open('./data/finetunev2.csv', 'r') as f:
    data_line = f.readlines()[1:]

mol_list = []
label_list = []
for data in data_line:
    smiles, label = data.strip('\n').split(',')
    mol = Chem.MolFromSmiles(smiles)
    mol_list.append(mol)
    label = int(label)
    label_list.append(label)
index = np.where(np.array(label_list) == 1)
print('Active contain: %d' % (index[0].shape[0]))
print('Non active contain: %d' % (len(mol_list) - (index[0].shape[0])))

'''5种分子表征形式：不同分子描述符'''
repre_list = ['ECFP', 'FCFP', 'Rdkit', 'MACCS', 'Avalon']
s = repre_list[3]  # 修改部分 一一对应

# _, fp_bit_list = mol_to_ecfp4(mol_list, 1024)
# _, fp_bit_list = mol_to_fcfp4(mol_list, 1024)
# _, fp_bit_list = mol_to_fp(mol_list, 1024)
_, fp_bit_list = mol_to_maccs(mol_list)
# _, fp_bit_list = mol_to_Avalon(mol_list, 1024)  # Avalon Fingerprint
x_data = np.array(fp_bit_list)
y_data = np.array(label_list)
print(mol_list)
print(len(fp_bit_list[0]))

CV_num = 5
KF = KFold(n_splits=CV_num, shuffle=True, random_state=0)  #

train_accuracy_all = 0
test_accuracy_all = 0

train_f1_score_all = 0
test_f1_score_all = 0

train_mcc_all = 0
test_mcc_all = 0

train_precision_all = 0
test_precision_all = 0

train_recall_all = 0
test_recall_all = 0

train_aucall = 0
test_aucall = 0

'''三类不同特征选择方法'''
# # 3.1.1利用方差进行特征选择,基于Filter的方法需要指定参数的大小
# from sklearn.feature_selection import VarianceThreshold
# VT = VarianceThreshold(threshold=0.01)  # origin(0.005)
# x_data_new = VT.fit_transform(x_data)
# print("VT picked " + str(np.size(x_data_new)) +
#       " variables and eliminated the other " + str(np.size(x_data) - np.size(x_data_new)) + " variables")
# x_data = x_data_new


# # 3.1.4 MI
# from sklearn.feature_selection import mutual_info_classif
# # discrete_features='auto',n_neighbors=20, copy=True, random_state=0
# MI_score = mutual_info_classif(x_data, y_data, discrete_features='auto',
#                                n_neighbors=20, copy=True, random_state=0)
# index = np.where(MI_score > 0.01)[0]  # 0.01
# print("MI picked " + str(np.size(index)) +
#       " variables and eliminated the other " + str(len(x_data[1]) - np.size(index)) + " variables")
# x_data = x_data[:, index]


# # 3.3.3利用Lasso进行特征选择
from sklearn.linear_model import LassoCV
import pickle
import joblib

# alpha_coef = list(np.arange(0.0001, 0.02, 0.0001))  # set alpha coefficient(0.0001, 0.0025, 0.0001)Lasso_model.alpha_ = 0.164
# Lasso_model = LassoCV(alphas=alpha_coef).fit(x_data, y_data)
# # Lasso_model.alpha_ = 0.164
# # Lasso_model.fit(x_data, y_data)
# # joblib.dump(Lasso_model, 'Lasso_model.pkl')
# # Lasso_model = joblib.load('Lasso_model.pkl')
#  # the best lasso model
# alpha_best = Lasso_model.alpha_
# print('Lasso_model_Best alpha is:%.4f' % alpha_best)
# index = np.nonzero(Lasso_model.coef_)[0]  # extract feature index
# print("Lasso picked " + str(np.size(index)) +
#       " variables and eliminated the other " + str(len(x_data[1]) - np.size(index)) + " variables")
# x_data = x_data[:, index]  # train feature




# #
alpha_coef = list(np.arange(0.003, 0.004, 0.0001))  # set alpha coefficient
Lasso_model = LassoCV(alphas=alpha_coef).fit(x_data, y_data)
# Lasso_model = joblib.load('Lasso_model.pkl')
alpha_best = Lasso_model.alpha_  # the best lasso model
print('Lasso_model_Best alpha is:%.4f' % alpha_best)
index = np.nonzero(Lasso_model.coef_)[0]  # extract feature index
print("Lasso picked " + str(np.size(index)) +
      " variables and eliminated the other " + str(len(x_data[1]) - np.size(index)) + " variables")
x_data = x_data[:, index]  # train feature


workbook = xlwt.Workbook()
sheet = workbook.add_sheet('Exp_results')
sheet.write(0, 1, 'Train_ACC')
sheet.write(0, 2, 'Test_ACC')
sheet.write(0, 3, 'Train_F1')
sheet.write(0, 4, 'Test_F1')
sheet.write(0, 5, 'Train_MCC')
sheet.write(0, 6, 'Test_MCC')

for i, (train_index, test_index) in enumerate(KF.split(x_data, y_data)):
    train_x, train_y = x_data[train_index], y_data[train_index]
    test_x, test_y = x_data[test_index], y_data[test_index]
    # # np.savetxt('text%d' % i, test_x)
    # # np.savetxt('train%d' % i, train_x)
    # train = np.loadtxt('train%d' % i)
    # test = np.loadtxt('text%d' % i)
    # print((test_x == test).all(), 'test')
    # print((test_x == test).all(), 'train')

    '''不同建模方法：RF、SVM、GBDT'''  # 修改模型参数
    '''RF_Classifier_model'''
    if save_data_path[7:-1] == 'RF_Results':
        from sklearn.ensemble import RandomForestClassifier
        from hyperopt import fmin, tpe, hp, partial, anneal


        def RF(argsDict):
            max_depth = argsDict["max_depth"] + 5  # [5, 20]
            n_estimators = argsDict['n_estimators'] * 5 + 50  # [50, 100]
            max_features = argsDict['max_features'] * 5 + 5  # [5, 30]
            global train_x, train_y
            RF_model = RandomForestClassifier(n_estimators=n_estimators,
                                              max_features=max_features,
                                              max_depth=max_depth,
                                              random_state=1)

            RF_model.fit(train_x, train_y)
            pre_train_y = RF_model.predict(train_x)
            pre_test_y = RF_model.predict(test_x)
            train_accuracy = accuracy_score(train_y, pre_train_y)
            test_accuracy = accuracy_score(test_y, pre_test_y)
            ave_accuracy = (train_accuracy + test_accuracy) / 2
            # metric = cross_val_score(model, train_x, train_y, cv=5, scoring="roc_auc").mean()
            # print(train_accuracy, test_accuracy)
            return -ave_accuracy


        space = {"max_depth": hp.randint("max_depth", 15),
                 "n_estimators": hp.randint("n_estimators", 10),  # [0,1,2,3,4,5] -> [50,]
                 "max_features": hp.randint("max_features", 6)}  # [0,1,2,3,4,5,] -> [50,]}

        algo = partial(tpe.suggest, n_startup_jobs=1)
        best = fmin(RF, space, algo=algo, max_evals=50)
        best = {'max_depth': 13, 'max_features': 1, 'n_estimators': 4}
        print(best)
        print(RF(best))

        max_depth = best["max_depth"] + 5  # [5, 20]
        n_estimators = best['n_estimators'] * 5 + 50  # [50, 100]
        max_features = best['max_features'] * 5 + 5  # [5, 30]

        RF_model = RandomForestClassifier(n_estimators=n_estimators,
                                          max_features=max_features,
                                          max_depth=max_depth,
                                          random_state=0)

        RF_model.fit(train_x, train_y)
        pre_train_y = RF_model.predict(train_x)
        pre_test_y = RF_model.predict(test_x)

        predict_prob_y = RF_model.predict_proba(test_x)[:, 1]
        train_prob_y = RF_model.predict_proba(train_x)[:, 1]
        # print(predict_prob_y.shape)  # 基于SVM对验证集做出预测，prodict_prob_y 为预测的概率
        # end svm ,start metrics
        roc_auc = sklearn.metrics.roc_auc_score(test_y, predict_prob_y)
        train_roc_auc = sklearn.metrics.roc_auc_score(train_y, train_prob_y)# 验证集上的auc值
        print('测试集AUC值', roc_auc)
        print('训练集AUC值', train_roc_auc)

        # predict_prob_y = RF_model.decision_function(test_x)
        # # roc_auc = dict()
        # fpr, tpr, threshold = roc_curve(test_y, predict_prob_y)
        # roc_auc = auc(fpr, tpr)
        # print('roc_auc:', roc_auc)

    elif save_data_path[7:-1] == 'SVM_Results':
        # '''SVM_classifier_Model'''
        from sklearn.svm import SVC
        from hyperopt import fmin, tpe, hp, partial, anneal
        import pickle


        def SVM(argsDict):
            C = argsDict['C']
            shrinking = argsDict['shrinking']
            kernel = argsDict['kernel']
            gamma = argsDict['gamma']
            global train_x, train_y

            SVM_Model = SVC(random_state=0,
                            C=C,
                            shrinking=shrinking,
                            gamma=gamma,
                            kernel=kernel,
                            probability=True,
                            )

            SVM_Model.fit(train_x, train_y)
            pre_train_y = SVM_Model.predict(train_x)
            pre_test_y = SVM_Model.predict(test_x)
            train_accuracy = accuracy_score(train_y, pre_train_y)
            test_accuracy = accuracy_score(test_y, pre_test_y)
            ave_accuracy = (train_accuracy + test_accuracy) / 2
            # ave_accuracy = (train_accuracy + test_accuracy) / 2
            return -ave_accuracy


        space = {'C': hp.uniform('C', 0.001, 1000),
                 'shrinking': hp.choice('shrinking', [True, False]),
                 'kernel': hp.choice('kernel', ['rbf', 'sigmoid', 'poly']),
                 'gamma': hp.choice('gamma', ['auto', hp.uniform('gamma_function', 0.0001, 8)])}

        algo = partial(tpe.suggest, n_startup_jobs=1)
        best = fmin(SVM, space, algo=algo, max_evals=5)
        best = {'C': 28.358068226759997, 'gamma': 0, 'kernel': 1, 'shrinking': 0}

        print(best)


        gamma_para = ['auto', hp.uniform('gamma_function', 0.0001, 8)]
        kernel_para = ['rbf', 'sigmoid', 'poly']
        C = best['C']
        if best['gamma'] == 0:
            gamma = gamma_para[best['gamma']]
        else:
            gamma = best['gamma_function']
        kernel = kernel_para[best['kernel']]
        shrinking = best['shrinking']

        SVM_Model = SVC(random_state=0,
                        C=C,
                        shrinking=shrinking,
                        gamma=gamma,
                        kernel=kernel,
                        probability=True)
        joblib.dump(SVM_Model, 'SVM_Model.pkl')
        # SVM_Model = joblib.load('SVM_Model%d.pkl' % i)
        # print('C:', SVM_Model.C, 'gamma:', SVM_Model.gamma,
        #       'kernel:', SVM_Model.kernel, 'shrinking:', SVM_Model.shrinking, '1111111111111111111111')

        SVM_Model.fit(train_x, train_y)

        pre_train_y = SVM_Model.predict(train_x)
        pre_test_y = SVM_Model.predict(test_x)



        predict_prob_y = SVM_Model.decision_function(test_x)
        print(len(predict_prob_y))
        p1 = SVM_Model.predict_proba(test_x)[:, 1]
        train_prob_y = SVM_Model.decision_function(train_x)
        # print(predict_prob_y.shape)  # 基于SVM对验证集做出预测，prodict_prob_y 为预测的概率
        # end svm ,start metrics
        roc_auc = sklearn.metrics.roc_auc_score(test_y, predict_prob_y)
        train_roc_auc = sklearn.metrics.roc_auc_score(train_y, train_prob_y)# 验证集上的auc值
        print('测试集AUC值', roc_auc)
        print('训练集AUC值', train_roc_auc)

        # predict_prob_y = SVM_Model.decision_function(test_x)
        # # roc_auc = dict()
        # fpr, tpr, threshold = roc_curve(test_y, predict_prob_y)
        # roc_auc = auc(fpr, tpr)
        # print('roc_auc:', roc_auc)


    '''GBDT model'''
    if save_data_path[7:-1] == 'GBDT_Results':
        from sklearn.ensemble import GradientBoostingClassifier
        from hyperopt import fmin, tpe, hp, partial


        def GBDT(argsDict):
            max_depth = argsDict["max_depth"] + 5
            n_estimators = argsDict['n_estimators'] * 5 + 50
            learning_rate = argsDict["learning_rate"] * 0.02 + 0.05
            subsample = argsDict["subsample"] * 0.1 + 0.7
            # print("max_depth:" + str(max_depth))
            # print("n_estimator:" + str(n_estimators))
            # print("learning_rate:" + str(learning_rate))
            # print("subsample:" + str(subsample))
            # print("min_child_weight:" + str(min_child_weight))
            global train_x, train_y

            GBDT_model = GradientBoostingClassifier(random_state=0,
                                                    max_depth=max_depth,  # 最大深度
                                                    n_estimators=n_estimators,  # 树的数量
                                                    learning_rate=learning_rate,  # 学习率
                                                    subsample=subsample, )  # 采样数

            GBDT_model.fit(train_x, train_y)
            pre_train_y = GBDT_model.predict(train_x)
            pre_test_y = GBDT_model.predict(test_x)
            train_accuracy = accuracy_score(train_y, pre_train_y)
            test_accuracy = accuracy_score(test_y, pre_test_y)
            ave_accuracy = (train_accuracy + test_accuracy) / 2
            # metric = cross_val_score(model, train_x, train_y, cv=5, scoring="roc_auc").mean()
            # print(train_accuracy, test_accuracy)
            return -ave_accuracy


        space = {"max_depth": hp.randint("max_depth", 15),  # [0-14]-->[5, 19]
                 "n_estimators": hp.randint("n_estimators", 10),  # [0-9] -> [50,100]
                 "learning_rate": hp.randint("learning_rate", 6),  # [0-5] -> [0.05,0.06]
                 "subsample": hp.randint("subsample", 4)}  # [0-3] -> [0.7-1.0]

        algo = partial(tpe.suggest, n_startup_jobs=1)
        best = fmin(GBDT, space, algo=algo, max_evals=50)
        best = {'learning_rate': 5, 'max_depth': 6, 'n_estimators': 9, 'subsample': 0}
        print(best)
        print(GBDT(best))

        max_depth = best["max_depth"] + 5
        n_estimators = best['n_estimators'] * 5 + 50
        learning_rate = best["learning_rate"] * 0.02 + 0.05
        subsample = best["subsample"] * 0.1 + 0.7
        print("max_depth:" + str(max_depth))
        print("n_estimator:" + str(n_estimators))
        print("learning_rate:" + str(learning_rate))
        print("subsample:" + str(subsample))

        GBDT_model = GradientBoostingClassifier(random_state=0,
                                                max_depth=max_depth,  # 最大深度
                                                n_estimators=n_estimators,  # 树的数量
                                                learning_rate=learning_rate,  # 学习率
                                                subsample=subsample, )  # 采样数

        GBDT_model.fit(train_x, train_y)
        pre_train_y = GBDT_model.predict(train_x)
        pre_test_y = GBDT_model.predict(test_x)
        #
        # predict_prob_y = GBDT_model.predict_proba(test_x)[:, 1]
        # # print(predict_prob_y.shape)  # 基于SVM对验证集做出预测，prodict_prob_y 为预测的概率
        # # # end svm ,start metrics
        # roc_auc = sklearn.metrics.roc_auc_score(test_y, predict_prob_y)  # 验证集上的auc值
        # print('AUC值', roc_auc)
        predict_prob_y = GBDT_model.predict_proba(test_x)[:, 1]
        train_prob_y = GBDT_model.predict_proba(train_x)[:, 1]
        # print(predict_prob_y.shape)  # 基于SVM对验证集做出预测，prodict_prob_y 为预测的概率
        # end svm ,start metrics
        roc_auc = sklearn.metrics.roc_auc_score(test_y, predict_prob_y)
        train_roc_auc = sklearn.metrics.roc_auc_score(train_y, train_prob_y)  # 验证集上的auc值
        print('测试集AUC值', roc_auc)
        print('训练集AUC值', train_roc_auc)

    train_accuracy = accuracy_score(train_y, pre_train_y)
    test_accuracy = accuracy_score(test_y, pre_test_y)

    train_accuracy_all += train_accuracy
    test_accuracy_all += test_accuracy

    train_f1_score = f1_score(train_y, np.array(pre_train_y))
    test_f1_score = f1_score(test_y, np.array(pre_test_y))

    train_f1_score_all += train_f1_score
    test_f1_score_all += test_f1_score

    train_mcc = matthews_corrcoef(train_y, np.array(pre_train_y))
    test_mcc = matthews_corrcoef(test_y, np.array(pre_test_y))

    train_mcc_all += train_mcc
    test_mcc_all += test_mcc

    train_precision = precision_score(train_y, np.array(pre_train_y))
    test_precision = precision_score(test_y, np.array(pre_test_y))

    train_precision_all += train_precision
    test_precision_all += test_precision


    train_recall = recall_score(train_y, np.array(pre_train_y))
    test_recall = recall_score(test_y, np.array(pre_test_y))


    train_recall_all += train_recall
    test_recall_all += test_recall

    sheet.write(i + 1, 0, 'Model' + str(i))
    sheet.write(i + 1, 1, '%.2f%%' % (train_accuracy * 100))
    sheet.write(i + 1, 2, '%.2f%%' % (test_accuracy * 100))
    sheet.write(i + 1, 3, '%0.4f' % (train_f1_score))
    sheet.write(i + 1, 4, '%0.4f' % (test_f1_score))
    sheet.write(i + 1, 5, '%0.2f' % (train_mcc))
    sheet.write(i + 1, 6, '%0.2f' % (test_mcc))
    test_aucall += roc_auc
    train_aucall += train_roc_auc
    sheet.write(i + 1, 7, '%0.4f' % (roc_auc))

    print(classification_report(y_true=test_y, y_pred=pre_test_y))

    print('%s_model-train_accuracy: %0.2f%%' % (str(i), train_accuracy * 100))
    print('%s_model-test_accuracy: %0.2f%%' % (str(i), test_accuracy * 100))

    print('%s_model-train_f1_score: %0.4f' % (str(i), train_f1_score))
    print('%s_model-test_f1_score: %0.4f' % (str(i), test_f1_score))

    print('%s_model-mcc: %0.2f' % (str(i), train_mcc))
    print('%s_model-mcc: %0.2f' % (str(i), test_mcc))

    print('%s_model-precision: %0.2f' % (str(i), train_precision))
    print('%s_model-precision: %0.2f' % (str(i), test_precision))

    print('%s_model-recall: %0.2f' % (str(i), train_recall))
    print('%s_model-recall: %0.2f' % (str(i), test_recall))

    if i == CV_num - 1:
        sheet.write(i + 2, 0, 'average_all')
        sheet.write(i + 2, 1, '%.2f%%' % (train_accuracy_all / CV_num * 100))
        sheet.write(i + 2, 2, '%.2f%%' % (test_accuracy_all / CV_num * 100))
        sheet.write(i + 2, 3, '%0.4f' % (train_f1_score_all / CV_num))
        sheet.write(i + 2, 4, '%0.4f' % (test_f1_score_all / CV_num))
        sheet.write(i + 2, 5, '%0.2f' % (train_mcc_all / CV_num))
        sheet.write(i + 2, 6, '%0.2f' % (test_mcc_all / CV_num))
        sheet.write(i + 2, 7, '%0.8f' % (test_aucall / CV_num))
        sheet.write(i + 2, 8, '%0.8f' % (train_aucall / CV_num))
    a = []
    b = []
    c = []
    d = []
    for k1 in range(34):
        a.append('%+-0.2f' % test_y[k1])
        b.append('%+-0.2f' % pre_test_y[k1])
        c.append('%+-0.2f' % predict_prob_y[k1])
        d.append('%+-0.2f' % p1[k1])
    print(a)
    print(b)
    print(c)
    print(d)
print('train_accuracy_all: %0.2f%%' % (train_accuracy_all / CV_num * 100))
print('train_f1_score_all: %0.4f' % (train_f1_score_all / CV_num))
print('train_mcc_all: %0.3f' % (train_mcc_all / CV_num))


print('test_accuracy_all: %0.2f%%' % (test_accuracy_all / CV_num * 100))
print('test_f1_score_all: %0.4f' % (test_f1_score_all / CV_num))
print('test_mcc_all: %0.3f' % (test_mcc_all / CV_num))
print('test_auc_all: %0.8f' % (test_aucall / CV_num))
print('train_auc_all: %0.8f' % (train_aucall / CV_num))
print('this is', s, save_data_path[7:-1])

print('train_precision_all: %0.2f%%' % (train_precision_all / CV_num * 100))

print('test_precision_all: %0.2f%%' % (test_precision_all / CV_num * 100))

print('train_recall_all: %0.2f%%' % (train_recall_all / CV_num * 100))
print('test_recall_all: %0.2f%%' % (test_recall_all / CV_num * 100))

workbook.save(os.path.join(save_data_path, 'Results_%s.xls' % (s)))# 保存

# # ''' XGB model'''
from hyperopt import fmin, tpe, hp, partial
# import xgboost as xgb
from sklearn.model_selection import cross_val_score

# def XGB(argsDict):
#     max_depth = argsDict["max_depth"] + 5
#     n_estimators = argsDict['n_estimators'] * 5 + 50
#     learning_rate = argsDict["learning_rate"] * 0.02 + 0.05
#     subsample = argsDict["subsample"] * 0.1 + 0.7
#     min_child_weight = argsDict["min_child_weight"] + 1
#     # print("max_depth:" + str(max_depth))
#     # print("n_estimator:" + str(n_estimators))
#     # print("learning_rate:" + str(learning_rate))
#     # print("subsample:" + str(subsample))
#     # print("min_child_weight:" + str(min_child_weight))
#     global train_x, train_y
#
#     model = xgb.XGBClassifier(nthread=4,  # 进程数
#                               max_depth=max_depth,  # 最大深度
#                               n_estimators=n_estimators,  # 树的数量
#                               learning_rate=learning_rate,  # 学习率
#                               subsample=subsample,  # 采样数
#                               min_child_weight=min_child_weight,  # 孩子数
#                               max_delta_step=10,  # 10步不降则停止
#                               objective="binary:logistic")
#     model.fit(train_x, train_y)
#     pre_train_y = model.predict(train_x)
#     pre_test_y = model.predict(test_x)
#     train_accuracy = accuracy_score(train_y, pre_train_y)
#     test_accuracy = accuracy_score(test_y, pre_test_y)
#     ave_accuracy = (train_accuracy + test_accuracy) / 2
#     # metric = cross_val_score(model, train_x, train_y, cv=5, scoring="roc_auc").mean()
#     # print(train_accuracy, test_accuracy)
#     return -ave_accuracy
#
#
# space = {"max_depth": hp.randint("max_depth", 15),
#          "n_estimators": hp.randint("n_estimators", 10),  # [0,1,2,3,4,5] -> [50,]
#          "learning_rate": hp.randint("learning_rate", 6),  # [0,1,2,3,4,5] -> 0.05,0.06
#          "subsample": hp.randint("subsample", 4),  # [0,1,2,3] -> [0.7,0.8,0.9,1.0]
#          "min_child_weight": hp.randint("min_child_weight", 5)}
#
# algo = partial(tpe.suggest, n_startup_jobs=1)
# best = fmin(XGB, space, algo=algo, max_evals=50)
# print(best)
# print(XGB(best))
#
# max_depth = best["max_depth"] + 5
# n_estimators = best['n_estimators'] * 5 + 50
# learning_rate = best["learning_rate"] * 0.02 + 0.05
# subsample = best["subsample"] * 0.1 + 0.7
# min_child_weight = best["min_child_weight"] + 1
# print("max_depth:" + str(max_depth))
# print("n_estimator:" + str(n_estimators))
# print("learning_rate:" + str(learning_rate))
# print("subsample:" + str(subsample))
# print("min_child_weight:" + str(min_child_weight))
#
# model = xgb.XGBClassifier(nthread=4,  # 进程数
#                           max_depth=max_depth,  # 最大深度
#                           n_estimators=n_estimators,  # 树的数量
#                           learning_rate=learning_rate,  # 学习率
#                           subsample=subsample,  # 采样数
#                           min_child_weight=min_child_weight,  # 孩子数
#                           max_delta_step=10,  # 10步不降则停止
#                           objective="binary:logistic")
# model.fit(train_x, train_y)
# pre_train_y = model.predict(train_x)
# pre_test_y = model.predict(test_x)
