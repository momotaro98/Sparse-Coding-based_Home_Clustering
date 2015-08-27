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

##########作成ディレクトリ設定####################
dir_name = "cond_infer" # 変更箇所
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
#homes = ['home201','home202','home203','home205','home207','home208','home302','home303','home304','home307']
homes = [train_home, test_home]

########出力ファイル名#####################
csv_file_name = "cond_infer_302_302_fin_20"

#########---01SC or normalSC 選択設定###########
#use_01SC = False
use_01SC = True

#########学習期間 試験期間 設定##############
first_day_train = datetime.datetime(2014, 7, 31)
last_day_train = datetime.datetime(2014, 8, 31)

first_day_test = datetime.datetime(2014, 7, 31)
last_day_test = datetime.datetime(2014, 8, 31)
############設定推定日ID################
#predict_day_id = 5
predict_day_id = int(argvs[3]) if argc != 1 else 5 #デフォルト宅

##########設定取得間隔##########
interval_min = 5
time_max = 24 * 60 / interval_min

##########スパース係数###########
lam = 0.02

####基底ベクトル数設定#############
base_func_num = 20
print "base_func_num = " + str(base_func_num)

max_iteration = 30

#######パーセプトロン利用設定##########
#use_disagrigation = True
use_disagrigation = False
alpha = 0.0000001


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
		Y += dot( Bs[device],As[device] )
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
	global use_01SC
 
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
			print "zero_one_optimizeA"
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
		
		print "optimizeB "
		B = optimizeB(X, lam, base_num, A, B)
		diff = sparce_coding_object(X, A, B, lam)
		#print diff 

		#収束条件
		if diff_last * 0.97 <= diff:
			break

		diff_last = diff 

	#print B_old[:,0]
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
	global use_01SC
	print "optimizeAs"

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
			print diff 

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
		B = As[device]

		(base_num, day_id_max) = A.shape

		for p in range(base_num):
			for q in range(day_id_max):

				BA = zeros((dot(B, A[:,q])).shape)
				for device2 in As.keys():

					A_j = As[device2]
					B_j = Bs[device2]

					BA += dot(B_j, A_j[:,q])

				bab = dot(BA, B[:,p])

				if dot(B[:,p], Bp[:,p]) == 0:
					A[p][q] = 0
				else:
					A[p][q] = (dot(B[:,p], X_all[:,q]) - lam - bab + A[p][q] * dot(B[:,p], B[:,p])) / dot(B[:,p], B[:,p])

				if A[p][q] < 0:
					A[p][q] = 0
		As[device] = A

	return As

#################disagrigation用メソッド################

def convergence_check(Bs,Bs_last):
	print "convergence_check"
	val = 0

	for device in Bs.keys():
		tmp = Bs[device] -Bs_last[device]
		val += (tmp*tmp).sum()
	print val
	return val



def perceptron_update_Bs(X,As_org,As_hat,Bs):
	global alpha
	print "perceptron"
	Bs_ret = {}

	BA_hat = makeBA(Bs,As_hat)
	BA_org = makeBA(Bs,As_org)

	X_minus_BA_hat = X - BA_hat
	X_minus_BA_org = X - BA_org

	print "get perceptron"
	for device in Bs.keys():
		Bs_ret[device] = Bs[device] - alpha * ( dot(X_minus_BA_hat,As_hat[device].transpose()) - dot(X_minus_BA_org,As_org[device].transpose()) )

		
	
	return Bs_ret


def makeBA(As,Bs):

	flag = False

	for device in Bs.keys():
		if flag:
			Cs += dot(As[device], Bs[device])
		else:
			Cs = dot(As[device], Bs[device])
			flag = True
		
	return Cs


def renormalize_Bs(Bs):
	retB ={}

	for device in Bs.keys():
		B = Bs[device]
		(time_id_max,base_num) = B.shape

		for q in range(base_num):
			b_q = math.sqrt(dot(B[:,q],B[:,q]))
			for p in range(time_id_max):
				if b_q == 0:
					B[p,q] =0
				else:
					B[p,q] = 1/b_q * math.fabs(B[p,q])
		retB[device] = B

	return retB




def disagrigation(X,As_org,Bs,lam,max_iteration): #
	As_hat = deepcopy(As_org) 
	last_val = float("inf")
	counter = 0 
	while True:
		Bs_last = deepcopy(Bs)
		As_hat, diff = optimizeAs(X, Bs, As_hat, lam, max_iteration) #論文ではA^
		Bs = perceptron_update_Bs(X,As_org,As_hat,Bs)
		print "after_perceptron_update"
		print Bs
		#Bs = renormalize_Bs(Bs)
		#print "after_renormalize"
		print Bs

		val =  convergence_check(Bs,Bs_last)
		if val > last_val or counter >15:
			break

		last_val = val
		counter +=1

	return Bs

############################original#################
def extract_Bs_ave_vector(B):
	B_device = B["cond"]
	#B_device = B["els_device"]
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
	#B_device = B["els_device"]
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
	#A_device = A["cond"]
	A_device = A["els_device"]
	(base_num, day_id_max) = A_device.shape
	Ah = array(hsplit(A_device, day_id_max))
	Ahh = Ah[0]
	list = []
	for elem in range(base_num):
		list.append(float(Ahh[elem]))
	nplist = array(list)
	return nplist
####################main##############################




day_max_train = (last_day_train - first_day_train).days +2 
day_max_test = (last_day_test - first_day_test).days +2 


last_day = max(last_day_test,last_day_train)

energy_usage_device={}#X_iに相当


energy_usage_all ={}#X barに相当　テスト用
energy_usage_device_answer={}#答え合わせ用


#XはT*m Tは日の添字　mは時間id

############################### start loading csv files #################################
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
			power = float(itemList[2])
			els_power = total_power - power
			tempar = float(itemList[3])
			if power > 10000:
				power = 0

			device = "cond"
			els_device = "els_device"

			if day <= last_day_train and day >= first_day_train:
				if homename not in energy_usage_device:
					energy_usage_device[homename] = {} 
					if device not in energy_usage_device[homename]:
						energy_usage_device[homename][device] = {} 
						energy_usage_device[homename][els_device] = {}
				energy_usage_device[homename][device][mydatetime] = power
				energy_usage_device[homename][els_device][mydatetime] = els_power

			if day <= last_day_test and day >= first_day_test:
				if homename not in energy_usage_all:
					energy_usage_all[homename] = {}
					if mydatetime not in energy_usage_all[homename]:
						energy_usage_all[homename][mydatetime] = 0

				energy_usage_all[homename][mydatetime] = total_power 

				if homename not in energy_usage_device_answer:
					energy_usage_device_answer[homename] = {} 
					if device not in energy_usage_device_answer[homename]:
						energy_usage_device_answer[homename][device] = {} 
						energy_usage_device_answer[homename][els_device] = {}
				energy_usage_device_answer[homename][device][mydatetime] = power
				energy_usage_device_answer[homename][els_device][mydatetime] = els_power



		#day += datetime.timedelta(days=1)

############################### end loading csv files #################################


X_device = {} #train用
X_total_train = {} #perceptron用

#print energy_usage_device[train_home]

for homename in energy_usage_device.keys():
	X_all_train =zeros((time_max, day_max_train)) #X barに相当 test甩
	if not homename in X_device:
		X_device[homename] = {}
	for device in energy_usage_device[homename].keys():
		X_device_each= zeros((time_max,day_max_train)) #X barに相当

		for mydatetime in energy_usage_device[homename][device].keys():
			day_id =(mydatetime - first_day_train).days
			time_id = (mydatetime.hour *60 + mydatetime.minute ) / interval_min

			if not day_id == predict_day_id:
				X_device_each[time_id][day_id] = energy_usage_device[homename][device][mydatetime]
				X_all_train[time_id][day_id] += energy_usage_device[homename][device][mydatetime]

		X_device[homename][device] = X_device_each #X_i

	X_total_train[homename] = X_all_train


#print energy_usage_all[train_home]

X_total = {} #test用
for homename in energy_usage_all.keys():
	#X_all_test = zeros((time_max, day_max_test)) #test用
	X_all_test = zeros((time_max, 1)) #test用
	for mydatetime in energy_usage_all[homename].keys():

		day_id =(mydatetime - first_day_test).days  #1日後が0
		time_id = (mydatetime.hour * 60 + mydatetime.minute) / interval_min

		if day_id == predict_day_id:
			#X_all_test[time_id][day_id] = energy_usage_all[homename][mydatetime]
			X_all_test[time_id][0] = energy_usage_all[homename][mydatetime]

	X_total[homename] = X_all_test



X_device_answer = {} #答え合わせ用
for homename in energy_usage_device_answer.keys():
	if not homename in X_device_answer:
		X_device_answer[homename] = {}
	for device in energy_usage_device_answer[homename].keys():
		#X_device_each= zeros((time_max,day_max_test)) #X barに相当
		X_device_each= zeros((time_max, 1)) #X barに相当
		for mydatetime in energy_usage_device_answer[homename][device].keys():

			day_id =(mydatetime - first_day_test).days  
			time_id = (mydatetime.hour * 60 + mydatetime.minute ) / interval_min

			if day_id == predict_day_id:
				#X_device_each[time_id][day_id] = energy_usage_device_answer[homename][device][mydatetime]
				X_device_each[time_id][0] = energy_usage_device_answer[homename][device][mydatetime]

		X_device_answer[homename][device] = X_device_each #X_i




Bs={}
#Base_device  R^T*n T= time_max  nはbaseの数 Tの添字のベクトルに対して二乗が<= 1

As_train={}
#Coefficient_device R^n*m nはbaseの数 m =day_max

#X= Base_device * Coefficient_device time_max*day_max
# today()メソッドで現在日付・時刻のdatetime型データの変数を取得


#f = open(path+'/result.csv', 'w') # 書き込みモードで開く

for homename, X_home_device in X_device.iteritems():
	if homename == train_home:
		#print homename
		if not homename in As_train:
			As_train[homename] = {}
		if not homename in Bs:
			Bs[homename] = {}
		for device, X_device_each in X_home_device.iteritems():#最適化する
			print device 
			(As_train[homename][device], Bs[homename][device]) = optimize_train(X_device_each,lam,base_func_num,max_iteration)
print "optimize_train end"


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

"""
if use_disagrigation:
	print "disagrigation phase"
	Bs = disagrigation(X_all_train, As_train,Bs, lam, max_iteration)

	'''
	Bs_ave_vec_after_perceptron = extract_Bs_ave_vector(Bs)
	plt.plot(Bs_ave_vec_after_perceptron)
	plt.show()
	plt.savefig(path + '/' + "B_af_perceptron.png")
	'''
	#各基底ベクトル
	Bs_each_vec_after_perceptron = extract_Bs_each_vector(Bs)
	count2 = 1
	for list in Bs_each_vec_after_perceptron:
		plt.plot(list)
		plt.show()
		plt.savefig(path + '/' +  "B_af_perceptron_" + str(count2) +".png")
		count2 += 1

	#plt.xlim(0, 288)
"""

print "study phase end"

'''
#自分の家の推定
for homename, X_home_total_test in X_total.iteritems():
	print As_test
	print homename
	As = initializeAs(X_home_total_test, Bs)
	As_test = optimizeAs(X_home_total_test, Bs, As, lam, max_iteration)
'''

for homename, Bs_device in Bs.iteritems():
	As = initializeAs(X_total[test_home], Bs_device)
	As_test, diff = optimizeAs(X_total[test_home], Bs_device, As, lam, max_iteration)

	'''
	print homename
	print "This home total difference value is..."
	print diff
	Y = dot(Bs_device['cond'], As_test['cond'])
	Z = Y - X_device_answer[homename]['cond']
	Z_double = Z * Z
	z_sum = Z_double.sum()
	print "This home cond diffrence value is..."
	print z_sum
	'''


'''
###As可視化######
As_vec_after_optimize = extract_As_vector(As_test)
#print As_vec_after_optimize
plt.plot(As_vec_after_optimize, '--o')
#plt.xlim(0, 200)
plt.show()  
plt.savefig(path + '/' + "A_af_optimize.png")
'''

##########################残りは出力########################

#答え合わせ
f2 = open(path + '/' + csv_file_name + '.csv', 'a') # 書き込みモードで開く

X_cond_answer_each = X_device_answer[test_home]['cond']
X_total_answer_each = X_total[test_home]

#出力
Y = dot(Bs[train_home]['cond'], As_test['cond'])
(time_id_max_,day_id_max_) = X_cond_answer_each.shape
for q in range(day_id_max_):
	for p in range(time_id_max_):
		f2.write(",".join([str(train_home),str(test_home),str(predict_day_id),str(p),str(X_total_answer_each[p][q]),str(X_cond_answer_each[p][q]),str(Y[p][q])]) + "\n")

'''
for device in sorted(X_device_answer.keys()):#最適化する

	X_device_answer_each = X_device_answer[device]

	#出力
	Y = dot(Bs[device], As_test[device])
	(time_id_max_,day_id_max_) = X_device_answer_each.shape
	for q in range(day_id_max_):
		for p in range(time_id_max_):
			f2.write(",".join([str(device),str(p),str(q),str(X_device_answer_each[p][q]),str(Y[p][q])]) + "\n")
'''

'''


#答え合わせ2
f3 = open(path+'/result_all2.csv', 'w') 
#出力
Z = zeros(X_all_test.shape)

for device in As_test.keys():
	Z += dot(Bs[device],As_test[device])

(time_id_max_,day_id_max_) =X_all_test.shape
for q in range(day_id_max_):
	for p in range(time_id_max_):
		f3.write(",".join([str(p),str(q),str(X_all_test[p][q]),str(Z[p][q])]) + "\n")



acc = accuracy(X_device_answer,Bs,As_test)*100
print acc
print path
print "base_func_num = " + str(base_func_num)


#答え合わせ2
f4 = open(path+'/config.txt', 'w') 

f4.write("01 sparce coding\n")
f4.write(str(base_func_num) + "\n")
#f4.write(str(folder)+ "\n")
#f4.write(str(tap_list)+ "\n")

f4.write(str(first_day_train)+ "\n")
f4.write(str(last_day_train)+ "\n")
f4.write(str(first_day_test)+ "\n")
f4.write(str(last_day_test)+ "\n")
f4.write(str(use_disagrigation)+ "\n")
#f4.write(str(choose_tap)+ "\n")
#f4.write(str(choose_tap_inverse)+ "\n")

f4.write(str(acc)+ "\n")

f4.write(str(path))

'''
