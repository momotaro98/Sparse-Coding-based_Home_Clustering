#!/usr/bin/env python
# -*- coding: utf-8 -*-
import datetime
import math
from copy import * 
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from pylab import *

from numpy import *

"""
How to Run
-------------------------------------------------
$ python SC_process.py home201 home202 5
-------------------------------------------------
"""

##########作成ディレクトリ設定####################
dir_name = "output_csv" # 変更箇所
d = datetime.datetime.today()
path = dir_name + '_%s-%s-%s_%s-%s' % (d.year, d.month, d.day,d.hour,d.minute)
#path = dir_name
os.mkdir(path)

argvs = sys.argv
argc = len(argvs)
########対象宅設定############
train_home = argvs[1] if argc != 1 else "home201"#デフォルト宅
test_home = argvs[2] if argc != 1 else "home202"#デフォルト宅

############homeディレクトリ####################
#homes = ['home201','home202','home203','home204','home205','home206','home207','home208','home301','home302','home303','home304','home305','home306','home307','home308']
homes = ['home201','home202','home203','home205','home207','home208','home302','home303','home304','home307']
#homes = [train_home, test_home]

########出力ファイル名#####################
csv_file_name = "output"

#########---01SC or normalSC 選択設定###########
#use_01SC = False
#use_01SC = True

#########学習期間 試験期間 設定##############
first_day_train = datetime.datetime(2014, 7, 31)
last_day_train = datetime.datetime(2014, 8, 31)

first_day_test = datetime.datetime(2014, 7, 31)
last_day_test = datetime.datetime(2014, 8, 31)
############設定推定日ID################
#predict_day_id = 5
predict_day_id = int(argvs[3]) if argc != 1 else 5 #デフォルト推定日

##########設定取得間隔##########
interval_min = 5
time_max = 24 * 60 / interval_min

##########スパース係数###########
lam = 0.02

####基底ベクトル数設定#############
base_func_num = 20

max_iteration = 30


############### 設定値出力 ########################
print("スパース係数:" + str(lam))
print("基底ベクトル:" + str(base_func_num))


###############評価関数用メソッド##############

def accuracy(Xs,Bs,As):
    ret1 = 0 
    ret2 = 0

    for device in As.keys():
        X = Xs[device]
        A = As[device]
        B = Bs[device]
        (time_id_max,day_id_max) = X.shape 
        Y = dot(B,A)

        for q in range(day_id_max):
            x = 0
            y = 0
            for p in range(time_id_max):
                x += X[p][q]
                y += Y[p][q]
                ret2 += X[p][q]

            ret1 += min(x,y)
    return ret1 / ret2 

def sparce_coding_object(X,A,B,lam): 
    Y = X - dot(B,A)
    Z =Y*Y
    z_sum = Z.sum() 
    ret = 0.5*z_sum + lam *  A.sum()

    return ret

def optimation_object(X_all,As,Bs,lam): 
    Y = 0 * X_all
    ret = 0

    for device in As.keys():
        Y += dot(Bs[device], As[device])
        ret += lam * As[device].sum()

    Z = X_all - Y 
    ret += (Z*Z).sum() 

    return ret 


############space coding用メッソド######################
def flip(x):
    if x==0:
        return 1
    else:
        return 0



def zero_one_optimizeA(X,lam,A,B):
    (base_num,day_id_max) = A.shape #nこのベース関数の係数、これは週による。
    last_value = sparce_coding_object(X,A,B,lam)

    for p in range(base_num):
        for q in range(day_id_max):

            A_pq_old = A[p][q] 

            A[p][q] = flip(A_pq_old)

            next_value = sparce_coding_object(X,A,B,lam)

            if next_value > last_value:
                A[p][q] = A_pq_old #もとに戻す。
            else:
                last_value = next_value

    return A

def normal_optimizeA(X, lam, A, B):
    (base_num, day_id_max) = A.shape

    for p in range(base_num):
        for q in range(day_id_max):
            if dot(B[:,p], B[:,p]) == 0:

                if -dot(B[:,p], X[:,q]) + lam + dot(dot(B.transpose()[p], B), A[:,q]) <= 0:
                    A[p][q] = 0
                else:
                    pass

            else:
                A[p][q] = (dot(B[:,p], X[:,q]) - lam - dot(dot(B.transpose()[p], B), A[:,q]) + A[p][q] * dot(B[:,p], B[:,p])) / dot(B[:,p], B[:,p])

            if A[p][q] <= 0:
                A[p][q] = 0

    return A


def optimizeB(X, lam, base_num, A, B):
    (time_id_max, base_num) = B.shape #n個のベース関数(これは週によらない。->つまり週をたくさん入れると学習される。)

    for p in range(time_id_max):
        for q in range(base_num):

            if dot(A[q], A[q]) == 0:
                if -(dot(X[p],A[q]) + dot(dot(B[p],A), A[q]) ) <= 0:
                    B[p][q] = 0
                else:
                    B[p][q] = math.sqrt(math.fabs(1- dot(B[:,q],B[:,q]) + B[p][q] *B[p][q]))
                #if p == 280 and q == 0:
                    #print "B[:,0]"
                    #print B[:,0]
            else:
                B[p][q] = (dot(X[p],A[q]) - dot(dot(B[p],A), A[q]) + B[p][q]*dot(A[q],A[q]) )/ dot(A[q],A[q])
                #if p == 280 and q == 0:
                    #print "B[:,0]"
                    #print B[:,0]

                #二項目 B[p][c]*A[c][b]*A[q][b]
                #三項目 B[p][q]*A[q][b]*A[q][b] )


            if B[p][q] <0:
                B[p][q] = 0
            

    return B

def initialize_A(size):
    #A = ones(size)
    A = zeros(size)
    #A = random.randint(0, 2, size)
    return A

def initialize_B(size):
    #B= random.random_sample(size) 
    B = ones(size) / 1000
    #B = ones(size)
    #Bはtime_idに対して、正規化する。
    return B

def optimize_train(X,lam,base_num,max_iteration): #
    #global use_01SC
    use_01SC = True
    #use_01SC = False
 
    (time_id_max,day_id_max) = X.shape 

    A = initialize_A((base_num,day_id_max))
    B = initialize_B((time_id_max,base_num))


    diff_last = float("inf")
    A_old = A.copy()
    B_old = B.copy()

    counter =0
    while counter <max_iteration:
        counter += 1

        A_old = A.copy()

        if use_01SC:
            #print "zero_one_optimizeA"
            A = zero_one_optimizeA(X, lam, A, B)
            #print A[:,20]
            #print A[:,25]
            #print A[:,30]
        else:
            #print "normal_optimizeA"
            A = normal_optimizeA(X, lam, A, B)
            #print A[:,20]
            #print A[:,25]
            #print A[:,30]
        '''
        diff = sparce_coding_object(X, A, B, lam)
        print diff


        #収束条件
        if diff_last * 0.97 <= diff:
            break

        diff_last =diff 
        '''

        B_old = B.copy()
        
        #print "optimizeB "
        B = optimizeB(X, lam, base_num, A, B)
        diff = sparce_coding_object(X, A, B, lam)
        #print diff 

        #収束条件
        if diff_last * 0.97 <= diff:
            break

        diff_last = diff 

    return (A_old,B_old)


#################dtest用メソッド################


def initializeAs(X_all,Bs):
    (time_id_max,day_id_max) = X_all.shape 
    As = {}
    for device in Bs.keys():

        (time_id_max,base_num) = Bs[device].shape #(week_max,2016)
        As[device] = initialize_A((base_num,day_id_max))
    return As



def optimizeAs(X_all, Bs, As, lam,max_iteration):
    use_01SC = False
    #use_01SC = True

    (time_id_max, day_id_max) = X_all.shape 

    diff_last = float("inf")

    counter = 0
    while counter < max_iteration :
        counter += 1

        As_old = deepcopy(As)

        try:
            if use_01SC:
                As = zero_one_optimizeAs_iteration(X_all, As, Bs, lam)
            else:
                As = normal_optimizeAs_iteration(X_all, As, Bs, lam)
            diff = optimation_object(X_all,As,Bs,lam)
            #print diff 

        except KeyboardInterrupt:
            As = As_old #戻す
            break
        
        #収束条件
        if diff_last*0.97 <= diff:
            break

        diff_last =diff 


    return As, diff


def zero_one_optimizeAs_iteration(X_all,As,Bs,lam):

    for device in As.keys():

        (base_num,day_id_max) = As[device].shape #(week_max,2016)

        for p in range(base_num):
            for q in range(day_id_max):
                last_value = optimation_object(X_all,As,Bs,lam)

                A_pq_old = As[device][p][q] 

                As[device][p][q] = flip(A_pq_old)

                next_value = optimation_object(X_all,As,Bs,lam)

                if next_value >= last_value:
                    As[device][p][q] = A_pq_old #もとに戻す。
     

    return As


def normal_optimizeAs_iteration(X_all, As, Bs, lam):

    for device in As.keys():

        A = As[device]
        B = Bs[device]

        (base_num, day_id_max) = A.shape

        for p in range(base_num):
            for q in range(day_id_max):

                BA = zeros((dot(B, A[:,q])).shape)
                for device2 in As.keys():

                    A_j = As[device2]
                    B_j = Bs[device2]

                    BA += dot(B_j, A_j[:,q])

                bab = dot(BA, B[:,p])

                if dot(B[:,p], B[:,p]) == 0:
                    A[p][q] = 0
                else:
                    A[p][q] = (dot(B[:,p], X_all[:,q]) - lam - bab + A[p][q] * dot(B[:,p], B[:,p])) / dot(B[:,p], B[:,p])

                if A[p][q] < 0:
                    A[p][q] = 0
        As[device] = A

    return As


############################ モデル（行列）の可視化用の関数  My Original Function #################
def extract_Bs_ave_vector(B):
    B_device = B["cond"]
    (time_id_max, base_num) = B_device.shape    
    Bh = array(hsplit(B_device, base_num))
    sum_vec = zeros((time_id_max, 1))
    for i in range(base_num):
        sum_vec += Bh[i]
    ave_vec = sum_vec / base_num
    list = []
    for elem in range(base_num):
        list.append(float(ave_vec[elem]))
    nplist = array(list)
    return nplist

def extract_Bs_each_vector(B):
    B_device = B["cond"]
    (time_id_max, base_num) = B_device.shape    
    Bh = array(hsplit(B_device, base_num))
    list2 = []
    for i in range(base_num):
        list = []
        Bhh = Bh[i]
        for j in range(time_id_max):
            list.append(float(Bhh[j]))
        list2.append(list)
    nplist = array(list2)
    return nplist

def extract_As_vector(A):
    A_device = A["cond"]
    (base_num, day_id_max) = A_device.shape
    Ah = array(hsplit(A_device, day_id_max))
    Ahh = Ah[0]
    list = []
    for elem in range(base_num):
        list.append(float(Ahh[elem]))
    nplist = array(list)
    return nplist




#################### Main Start ##############################


day_max_train = (last_day_train - first_day_train).days +2 
day_max_test = (last_day_test - first_day_test).days +2 


last_day = max(last_day_test,last_day_train)

energy_usage_device={} # 空調電力格納用
energy_usage_device_answer={} # 答え合わせ期間格納用


#XはT*m Tは日の添字 mは時間id

############################### start loading csv files #################################
print("Start CSV Loading")
#ファイル読み込み
for homename in homes:
    day = min(first_day_test,first_day_train)
    while day <= last_day:
        day += datetime.timedelta(days=1)
        homename_folder = homename
        filename =  day.strftime(homename + '_%Y-%m-%d.csv')
        #print filename

        for dpath, dnames, fnames in os.walk('./csv_files/' + homename_folder):
            if filename in fnames:
                break
        else:
            continue

        for line in open('./csv_files/'+homename_folder+'/'+ filename, 'r'):
            itemList = line[:-1].split(',')
            mydatetime = datetime.datetime.strptime(itemList[0],'%Y-%m-%d %H:%M:00+09:%S')

            datetime_str = mydatetime.strftime('%Y/%m/%d %H:%M:%S')
            total_power = float(itemList[1])
            cond_power = float(itemList[2])
            tempar = float(itemList[3])
            if total_power > 10000:
                total_power = 0
            if cond_power > 10000:
                cond_power = 0

            total = "total"
            cond = "cond"


            # 学習用 energy_usage_device
            # 総電力モデル作成用 energy_usage_device[homename][total]
            # 再構築用 energy_usage_device[既知2宅][cond]
            if day <= last_day_train and day >= first_day_train:
                if homename not in energy_usage_device:
                    energy_usage_device[homename] = {} 
                    if total not in energy_usage_device[homename]:
                        energy_usage_device[homename][total] = {} 
                        energy_usage_device[homename][cond] = {} 
                energy_usage_device[homename][total][mydatetime] = total_power
                energy_usage_device[homename][cond][mydatetime] = cond_power

            # 答え合わせ用 energy_usage_device_answer
            # 空調電力の答え 既知2宅以外宅
            if day <= last_day_test and day >= first_day_test:
                if homename not in energy_usage_device_answer:
                    energy_usage_device_answer[homename] = {} 
                    if cond not in energy_usage_device_answer[homename]:
                        energy_usage_device_answer[homename][cond] = {} 
                energy_usage_device_answer[homename][cond][mydatetime] = cond_power



        #day += datetime.timedelta(days=1)

############################### end loading csv files #################################



#print energy_usage_device[train_home]

######################### Sparse Coding処理のための変数定義 ####################

# 空調電力(再構築用) と 総電力(モデル作成用)
# 空調電力で必要なのは、空調電力既知の２宅分で、総電力ではそれ以外の宅
X_device = {}
for homename in energy_usage_device.keys():
    if not homename in X_device:
        X_device[homename] = {}
    for device in energy_usage_device[homename].keys():
        X_device_each= zeros((time_max,day_max_train)) # Numpy行列初期化

        for mydatetime in energy_usage_device[homename][device].keys():
            day_id =(mydatetime - first_day_train).days
            time_id = (mydatetime.hour *60 + mydatetime.minute ) / interval_min

            X_device_each[time_id][day_id] = energy_usage_device[homename][device][mydatetime]

        X_device[homename][device] = X_device_each

# 答え合わせ用空調電力
# 必要なのは、空調電力が未知の家庭分
# 答え合わせでは、１日分とする
X_device_answer = {} #答え合わせ用
for homename in energy_usage_device_answer.keys():
    if not homename in X_device_answer:
        X_device_answer[homename] = {}
    for device in energy_usage_device_answer[homename].keys():
        #X_device_each= zeros((time_max,day_max_test))
        X_device_each= zeros((time_max, 1))
        for mydatetime in energy_usage_device_answer[homename][device].keys():
            day_id =(mydatetime - first_day_test).days  
            time_id = (mydatetime.hour * 60 + mydatetime.minute ) / interval_min
            if day_id == predict_day_id:
                #X_device_each[time_id][day_id] = energy_usage_device_answer[homename][device][mydatetime]
                X_device_each[time_id][0] = energy_usage_device_answer[homename][device][mydatetime]

        X_device_answer[homename][device] = X_device_each #X_i

#################### SC処理開始 ################################

print("Start Creating Models")
############################# モデル作成処理(総電力モデル作成) ###################################
# 総電力モデルであるBsを求める
Bs = {}
As_train = {}
As_train[train_home] = {}
Bs[train_home] = {}

(As_train[train_home]["total"], Bs[train_home]["total"]) = optimize_train(\
                    X_device[train_home]["total"],\
                    lam,\
                    base_func_num,\
                    max_iteration\
                    )


print("End Creating Models")

print("Start Reconstructing")
############################### 再構築処理 #############################################
# 空調電力利用基底を表しているAs_testを求める
As_test = {}
As = initializeAs(X_device[test_home]["cond"], Bs[train_home])
As_test, diff = optimizeAs(X_device[test_home]["cond"], Bs[train_home], As, lam, max_iteration)
#print(As_test["total"]) # As_test["total"] に行列が格納されている
print(As_test["total"].sum()) # As_test["total"] に行列が格納されている

print("End Reconstructing")

#################### SC処理終了 ################################



##################### 残りは出力 ##############################

#答え合わせ
#CSV出力
f2 = open(path + '/' + csv_file_name + '.csv', 'a') # 書き込みモードで開く

# 実際の空調電力 X_cond_answer
X_cond_answer = X_device_answer[train_home]['cond']
# 推定の空調電力 Y
Y = dot(Bs[train_home]['total'], \
        As_test['total']) # 意味的に本当はAs_test['cond']である

(time_id_max_, day_id_max_) = X_cond_answer.shape
for p in range(time_id_max_):
    f2.write(",".join([\
        str(train_home), \
        str(test_home), \
        str(predict_day_id), \
        str(p), \
        str(X_cond_answer[p][0]), \
        str(Y[p][predict_day_id])]) \
        + "\n")



'''
#modelBの可視化
Bs_after_train_ave_vec = extract_Bs_ave_vector(Bs)
#print Bs_after_train_ave_vec
plt.plot(Bs_after_train_ave_vec)
#plt.xlim(0, 288)
plt.show()  
plt.savefig(path + '/' + "B_af_train.png")
'''

'''
#各基底ベクトル可視化
Bs_after_train_each_vec = extract_Bs_each_vector(Bs)
count = 1
for list in Bs_after_train_each_vec:
    plt.plot(list)
    plt.show()
    plt.savefig(path + '/' +  "B_af_train_" + str(count) +".png")
    count += 1
'''


'''
#modelAの可視化
As_after_train_vec = extract_As_vector(As_train)
plt.plot(As_after_train_vec)
#plt.xlim(0, 200)
plt.show()  
plt.savefig(path + '/' + "A_af_train.png")
'''
