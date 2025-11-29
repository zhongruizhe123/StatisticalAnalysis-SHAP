import pandas as pd
from sklearn import svm
import torch
import torch.nn as nn
import torch.optim as optim
import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn.functional as F
import pandas as pd
from torch.optim.lr_scheduler import StepLR
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import os
from sklearn.model_selection  import KFold
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection  import LeaveOneGroupOut
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes  import GaussianNB
from sklearn.tree  import DecisionTreeClassifier
# 读取Excel文件的第七个工作表
def standardize_shap_values(shap_obj):
    if hasattr(shap_obj, 'values'):  # 处理Explanation对象
        values = shap_obj.values
    else:  # 处理普通数组或列表
        values = shap_obj[1] if isinstance(shap_obj, list) else shap_obj

    standardized = (values - np.mean(values)) / np.std(values)
    return standardized
def GradientBoosting_model(X_train, X_test, y_train, y_test):
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    # model = RandomForestClassifier(n_estimators=100, random_state=42)
    # model = svm.SVC(kernel='linear')  # 你可以尝试不同的核函数，如'rbf', 'poly'等
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    y_pred_Probability = probabilities[:, 1]  # 形状 (43,)   变成1个值
    return y_pred_Probability, model
def LogisticRegression_model(X_train, X_test, y_train, y_test):
    model = LogisticRegression(penalty='l2', C=1.0, solver='liblinear', max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    y_pred_Probability = probabilities[:, 1]  # 形状 (43,)   变成1个值
    return y_pred_Probability, model
def RandomForest_model(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    y_pred_Probability = probabilities[:, 1]  # 形状 (43,)
    return y_pred_Probability, model
def SVM_model(X_train, X_test, y_train, y_test):
    model = svm.SVC(kernel='linear', probability=True)  # 你可以尝试不同的核函数，如'rbf', 'poly'等
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_Probability = model.predict_proba(X_test)
    y_pred_Probability = y_pred_Probability[:, 1]  # 形状 (43,)
    return y_pred_Probability, model
def xgboost_model(X_train, X_test, y_train, y_test):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    num_round = 100  # 迭代次数
    bst = xgb.train(params, dtrain, num_round)
    y_pred = bst.predict(dtest)
    return y_pred, bst
def GaussianNB_model(X_train, X_test, y_train, y_test):
    model = GaussianNB(var_smoothing=1e-9)  # 处理连续特
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_Probability = model.predict_proba(X_test)
    y_pred_Probability = y_pred_Probability[:, 1]  # 形状 (43,)
    return y_pred_Probability, model
def DecisionTreeClassifier_model(X_train, X_test, y_train, y_test):
    model = DecisionTreeClassifier(min_samples_leaf=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_Probability = model.predict_proba(X_test)
    y_pred_Probability = y_pred_Probability[:, 1]  # 形状 (43,)
    return y_pred_Probability, model
def check_invalid_values(data, name):
    if hasattr(data, 'values'):  # 如果是 pandas DataFrame/Series
        data = data.values
    print(f"检查 {name}:")
    print("  NaN 数量:", np.isnan(data).sum())
    print("  Inf 数量:", np.isinf(data).sum())
    print("  最大值:", np.nanmax(data))
    print("  最小值:", np.nanmin(data))

file_path = 'J:\Project\EEG_fMRI_Statistics\SHAP-CNN\EEG+fMRI输入数据new(174)-实验一-38.xlsx'  # 替换为你的Excel文件路径
sheet_name = 'Sheet1'  # 假设第七个工作表的名称是Sheet7，如果不是，请替换为实际名称
excel_no_values = [0,1,2,3,4,5,6,7]
model_code = """
params = {
    'objective': 'binary:logistic',  # 多分类任务
    'eta': 0.1,  # 学习率
    'max_depth': 6,  # 树的最大深度
    'subsample': 0.8,  # 子样本比例
    'colsample_bytree': 0.8,  # 每棵树随机采样的列的比例
    'eval_metric': 'logloss'  # 多分类对数损失，虽然这不是准确率，但可以作为训练时的监控指标
}
"""
params = {
    'objective': 'binary:logistic',  # 多分类任务
    'eta': 0.1,  # 学习率
    'max_depth': 6,  # 树的最大深度
    'subsample': 0.8,  # 子样本比例
    'colsample_bytree': 0.8,  # 每棵树随机采样的列的比例
    'eval_metric': 'logloss'  # 多分类对数损失，虽然这不是准确率，但可以作为训练时的监控指标
}
feature_name = ['DeltaTM_E->D', 'MeanGFP_A', 'MeanGFP_B', 'MeanGFP_C', 'MeanGFP_D', 'MeanGFP_E', 'GFPStdDev_A',
                'GFPStdDev_B', 'GFPStdDev_C', 'GFPStdDev_D', 'GFPStdDev_E', 'DefaultMode_Pearson correlation',
                'DefaultMode_Cosine similarity',
                'DefaultMode_Euclidean distance', 'SensoriMotor_Pearson correlation', 'SensoriMotor_Cosine similarity',
                'SensoriMotor_Euclidean distance',
                'Visual_Pearson correlation', 'Visual_Cosine similarity', 'Visual_Euclidean distance',
                'Salience_Pearson correlation',
                'Salience_Cosine similarity', 'Salience_Euclidean distance', 'DorsalAttention_Pearson correlation',
                'DorsalAttention_Cosine similarity',
                'DorsalAttention_Euclidean distance', 'FrontoParietal_Pearson correlation',
                'FrontoParietal_Cosine similarity',
                'FrontoParietal_Euclidean distance', 'Language_Pearson correlation', 'Language_Cosine similarity',
                'Language_Euclidean distance',
                'Cerebellar_Pearson correlation', 'Cerebellar_Cosine similarity', 'Cerebellar_Euclidean distance',
                'Whole-brain_PearsonCorrelation',
                'Whole-brain_CosineSimilarity', 'Whole-brain_EuclideanDistance'
                ]
print('len(feature_name)',len(feature_name))
age_i = [1,2,3,4]
# for age_i in [1,2,3,4]:        #1-5    年龄

############################################写全部模型##################################
model_list = ['GradientBoosting', 'LogisticRegression', 'RandomForest', 'SVM', 'xgboost', 'GaussianNB', 'DecisionTreeClassifier']

model_A = 'LogisticRegression'        #这里写model
model_B = 'GaussianNB'
folder_path_age = f"J:\\Project\\EEG_fMRI_Statistics\\SHAP-CNN\\new联合模型实验记录(Mann-Whitney)\\SHAP\\五折\\实验一\\{model_A}+{model_B}\\按年龄筛选数据-交叉验证-trick2-new-38\\"
if not os.path.exists(folder_path_age):
    os.makedirs(folder_path_age)
max_KFold_Accuracy = 0
############################################写全部模型##################################
for illness_i in [1]:        #0和5   疾病
    folder_path_illness = folder_path_age+'/疾病分类'
    if not os.path.exists(folder_path_illness):
        os.makedirs(folder_path_illness)
    max_random = 0
    max_Accuracy = 0
    Average_Accuracy = 0
    # Average_Accuracy_RandomForest = []
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    # print(df.iloc[:, 2].isin([0, illness_i]))
    # print((df.iloc[:, 6] == age_i) & (df.iloc[:, 2].isin([0, illness_i])))
    filtered_df = df[df.iloc[:, 2].isin([0, illness_i])]  # print('df.shape', df.shape)
    # print('df.shape', df.shape)
    print(filtered_df)
    # 假设第二列为需要预测的输出（目标变量），其余列都为输入（特征变量）
    print([i for i in range(df.shape[1]) if i not in excel_no_values])
    X = filtered_df.iloc[:, [i for i in range(df.shape[1]) if i not in excel_no_values]].values  # 读取除了第2列之外的所有列作为特征
    y = filtered_df.iloc[:, 2].values  # 读取第二列作为目标变量（输出）
    # print(X)
    # print(y)
    # print(type(y))
    y[y != 0] = 1
    count_ones = np.sum(y == 1)
    print("疾病个数：", count_ones)  # 输出: 1
    count_zero = np.sum(y == 0)
    print("健康个数：", count_zero)  # 输出: 1
    # print(y)
    # y = (y == 2 or y == 3 or y == 4 or y == 5 or y == 6).long()
    # 分割数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    print("训练集个数：", len(X_train))  # 159
    print("测试集个数：", len(X_test))  # 40
    check_invalid_values(X_train, "X_train")
    check_invalid_values(y_train, "y_train")
    # K = 0
    k_max = 0
    max_accuracy_all = 0
    max_random_all = 0          #    全局最佳的随机种子
    Average_Accuracy_all = 0   #随机种子的平均
    k_Average = 0
    n_subjects = len(y)  # 假设50个受试者
    subject_ids = np.repeat(np.arange(n_subjects), 1)
    logo = LeaveOneGroupOut()
    print("X", X.shape)
    print(len(y))
    Average_Accuracy_list = []
    KFold_Accuracy = 0
    K = 1.85

    max_KFold_Accuracy = 0
    max_random = 0
    KFold_Accuracy = 0
    for random_i in range(10):
        kf = KFold(n_splits=5, shuffle=True, random_state=random_i)  # 五折代码'

        # print("X.shape", X.shape)
        for fold_i, (train_idx, test_idx) in enumerate(kf.split(X)):
    # for random_i in range(30):
    # print(logo.split(X, y, groups=subject_ids))

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            # print('开始model',model_A,model_B)
            ##################################################写模型1的逻辑##################################
            if model_A == 'GradientBoosting':
                y_pred_A, ModelA = GradientBoosting_model(X_train, X_test, y_train, y_test)
            elif model_A == 'LogisticRegression':
                y_pred_A, ModelA = LogisticRegression_model(X_train, X_test, y_train, y_test)
            elif model_A == 'RandomForest':
                y_pred_A, ModelA = RandomForest_model(X_train, X_test, y_train, y_test)
            elif model_A == 'SVM':
                y_pred_A, ModelA = SVM_model(X_train, X_test, y_train, y_test)
            elif model_A == 'xgboost':
                y_pred_A, ModelA = xgboost_model(X_train, X_test, y_train, y_test)
            elif model_A == 'GaussianNB':
                y_pred_A, ModelA = GaussianNB_model(X_train, X_test, y_train, y_test)
            else:
                print("model_A:", model_A)
                continue
            ##################################################写模型1的逻辑##################################
            ##################################################写模型2的逻辑##################################
            if model_B == 'LogisticRegression':
                y_pred_B, ModelB = LogisticRegression_model(X_train, X_test, y_train, y_test)
            elif model_B == 'RandomForest':
                y_pred_B, ModelB = RandomForest_model(X_train, X_test, y_train, y_test)
            elif model_B == 'SVM':
                y_pred_B, ModelB  = SVM_model(X_train, X_test, y_train, y_test)
            elif model_B == 'xgboost':
                y_pred_B, ModelB  = xgboost_model(X_train, X_test, y_train, y_test)
            elif model_B == 'GaussianNB':
                y_pred_B, ModelB  = GaussianNB_model(X_train, X_test, y_train, y_test)
            elif model_B == 'DecisionTreeClassifier':
                y_pred_B, ModelB  = DecisionTreeClassifier_model(X_train, X_test, y_train, y_test)
            else:
                print("model_B:", model_B)
                continue
            ##################################################写模型2的逻辑##################################

            ########

            avg_arr = (K * y_pred_A + (2 - K) * y_pred_B) / 2
            # print('avg_arr', avg_arr)
            y_pred = (avg_arr >= 0.5).astype(int)
            accuracy = accuracy_score(y_test, y_pred)  # 混合的准确率
            KFold_Accuracy += accuracy  # 获得一个随机种子的五折的平均准确率
            if max_KFold_Accuracy < accuracy:
                max_KFold_Accuracy = accuracy
#######################################################################################################SHAP
                # probabilities = ModelA.predict_proba(X_test)
                # print(probabilities)
                print(len(X_test))
                #########################################在最佳里再训练一遍
                explainer_A = shap.KernelExplainer(ModelA.predict_proba, X_train)
                shap_values_A = explainer_A(X_test)

                explainer_B = shap.KernelExplainer(ModelB.predict_proba, X_train)
                shap_values_B = explainer_B(X_test)

                print(shap_values_A.shape, shap_values_B.shape)         #B会生成一个(35, 38, 2)的数组，我们只要[:, :, 1]，为1的shap
                #
                print("SHAP值特征维度:", shap_values_A.shape[1])  # 多分类时检查shap_values_A[0].shape
                print("X_test特征维度:", X_test.shape[1])
                print("feature_name长度:", len(feature_name))


                combined_shap = K * shap_values_A[:, :, 1] + (2 - K) * shap_values_B[:, :, 1]

                shap.summary_plot(combined_shap, X_test, max_display=38,
                                  feature_names=feature_name)  # 注意这里X_test[:10]仅用于摘要图的x轴标签，实际数据是shap_values对应的
                plt.savefig(folder_path_illness + '/' + str(age_i) + '_' + str(illness_i) + "_融合健康点图.png")
                plt.close()

                shap.summary_plot(combined_shap, X_test,
                                  plot_type="bar", max_display=38, feature_names=feature_name)
                plt.savefig(folder_path_illness + '/' + str(age_i) + '_' + str(illness_i) + "_融合健康条形图.png")
                plt.close()




#######################################################################################################SHAP
        # 计算准确率
        Average_Accuracy += KFold_Accuracy#获得五折十个随机种子的平均准确率（放在论文上的准确率）
        if max_KFold_Accuracy < KFold_Accuracy:
            max_KFold_Accuracy = KFold_Accuracy
            max_random = random_i                   #找到最佳的随机种子
        KFold_Accuracy = 0
    Average_Accuracy_list.append([K, Average_Accuracy/ 50, max_random]) #50代表的是五折5*十个随机种子10
    if Average_Accuracy_all < Average_Accuracy:
        k_Average = K
        Average_Accuracy_all = Average_Accuracy
    print(K, Average_Accuracy / 50, max_random)
    K = round(K + 0.01, 2)
    Average_Accuracy = 0
    # 写入CSV文件
    csvpath = os.path.join(folder_path_age,'output.csv')
    with open(csvpath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(Average_Accuracy_list)  # 注意使用writerows而不是writerow
    i0 = "年龄分类：" + str(age_i) + "   疾病分类：" + str(illness_i)
    i1 = "疾病个数：" + str(count_ones)
    i2 = "健康个数：" + str(count_zero)
    i3 = "训练集个数：" + str(len(X_train))
    i4 = "测试集个数：" + str(len(X_test))
    i8 = f"{len(y)}个平均准确率：" + str(Average_Accuracy_all / len(y)) + "%"
    i9 = "平均的平衡因子：" + str(k_Average)
    lines_to_write = [
        i0, i1, i2, i3, i4, i8, i9, "\n\n\n", model_code
    ]
    txt_illness = folder_path_illness + '/' + str(age_i) + '_' + str(illness_i) + '.txt'
    with open(txt_illness, "w", encoding="utf-8") as file:
        for line in lines_to_write:
            file.write(line + "\n")
#########################################在最佳里再训练一遍

    # plt.close()
#########################################在最佳里再训练一遍


# 假设第二列为需要预测的输出（目标变量），其余列都为输入（特征变量）

# 创建SVM模型并进行训练


# 如果你想要保存模型以便将来使用，可以使用joblib库
