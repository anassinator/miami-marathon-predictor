import matplotlib.pyplot as plt
import random
import numpy as np
current_year = 2016
# -----------------------------------------------------------------------------------------------------
#    GET RELEVANT FEATURES FROM YEARS 2015 AND LESS
f = open("../generated_data/pre2016.txt")
text = f.read()
lines = text.split("\n")

relevant_indices = [2, 4, 5, 7] #excluding ID

indexOf = {}

participant_data = []
for i in range(0,len(lines)):
    info = lines[i].split(",")
    relevant_features = []
    for r in relevant_indices:
        relevant_features.append(info[r])
    ID = int(info[0])
    if i==0:
        participant_data.append([ID])
        participant_data[-1].append([relevant_features])
        number_participants = 1
        continue
    if participant_data[-1][0] != ID:
        participant_data.append([ID])
        participant_data[-1].append([relevant_features])
        number_participants += 1
    else: 
        participant_data[-1][1].append(relevant_features) 
    
    indexOf[ID] = number_participants - 1
# -----------------------------------------------------------------------------------------------------
#    GET Y-LABELS FROM 2016 SET
f = open("../generated_data/2016.txt")
text = f.read()
lines = text.split("\n")

thisyear_participants = []
for line in lines:
    info = line.split(",")
    thisyear_participants.append(int(info[0]))
    
for i in range(0, len(participant_data)):
    if participant_data[i][0] in thisyear_participants:
        participant_data[i].append(1)
    else:
        participant_data[i].append(0)
        
#for p in participant_data:
    #print p
random.shuffle(participant_data)
# -----------------------------------------------------------------------------------------------------
trainingsize = 0.70
f = open("../generated_data/training_data.txt",'w')
for i in range(0,int(len(participant_data)*trainingsize)):
    if i == (int(len(participant_data)*trainingsize) -1):
        f.write("%s"%participant_data[i])
    else:
        f.write("%s\n"%participant_data[i])

f = open("../generated_data/testing_data.txt",'w')
for i in range(int(len(participant_data)*trainingsize), len(participant_data)):
    if i == (int(len(participant_data)*trainingsize)-1):
        f.write("%s"%participant_data[i])
    else:
        f.write("%s\n"%participant_data[i])
# -----------------------------------------------------------------------------------------------------
training_set = []
testing_set = []
for i in range(0,int(len(participant_data)*trainingsize)):
    training_set.append(participant_data[i])

for i in range(int(len(participant_data)*trainingsize), len(participant_data)):
    testing_set.append(participant_data[i])
# -----------------------------------------------------------------------------------------------------
def get_avgRuntime_inSec(listt):
    total = 0
    for l in listt:
        temp = l.split(":")
        total += int(temp[0])*60*60 + int(temp[1])*60 + int(temp[2])
    return (total+0.0)/len(listt)
# -----------------------------------------------------------------------------------------------------
# MAKE A LIST OF X VECTORS
X_list = []
Y_list = []
for participant in training_set: #switch this to testing set
    #get some info
    data_vectors = participant[1]
    number_times_participated = len(data_vectors)
    last_time_participated = 0
    age_lasttime = 0
    runtimes = []
    x = []
    
    for data in data_vectors:
        if int(data[-1]) > last_time_participated:
            last_time_participated = int(data[-1])
            age_lasttime = int(data[0])
        runtimes.append(data[2])
    
    #generate some info
    years_since_participated = current_year - last_time_participated
    age = age_lasttime + years_since_participated 
    #note, you might want to use age^2. Since as age increases, you tend be less likely to do physical activity in an expontential manner.
    avg_runtime = get_avgRuntime_inSec(runtimes)
    #print age, number_times_participated, years_since_participated, avg_runtime
    
    x.append(age)
    x.append(number_times_participated)
    x.append(years_since_participated)
    #x.append(avg_runtime)
    
    X_list.append(x)
    Y_list.append(participant[-1])

n = len(participant_data)
number_features = len(X_list[0])    
X=np.array([np.array(xi) for xi in X_list])
Y=np.array([np.array(yi) for yi in Y_list])
W_list = []
for i in range(0,number_features):
    W_list.append(0)
W=np.array([np.array(wi) for wi in W_list])
print "X = %s"%X
print "Y = %s"%Y
print "W = %s"%W
# -----------------------------------------------------------------------------------------------------
# BEGIN LOGISTIC REGRESSION
alpha = 0.01
number_iterations = 1000

def sigmoid(a): # The logistic function
    return (1 / (1 + np.exp(-a)))

cost_values = []
def gradient_descent(W):
    for iter_n in range(0, number_iterations):
    #while True:
        wTx = np.dot(X,np.transpose(W))
        difference =  np.transpose(sigmoid(wTx)) - Y
        n = X.shape[1]
        
        w = W.tolist()
        diff = difference.tolist()
        
        temp = []
        for j in range(0, n):
            summ = 0
            for i in range(0, X.shape[0]):
                summ += diff[i]*X[i,j]
            summ = summ/X.shape[0]
            temp.append(w[j] - (alpha * summ))
            
        for j in range(0,len(w)):
            w[j] = temp[j]
        W = np.array([np.array(wi) for wi in w])
        #print np.array([np.array(wi) for wi in w])
        print W

        wTx = sigmoid(X.dot(W))
        #
        c1 = (Y.dot(np.log(wTx)))
        c2 = (1 - Y).dot(np.log(1 - wTx))
        cost = (-1) * (c1 - c2) / X.shape[0]
        print np.sum(cost)
        cost_values.append(np.sum(cost))

         
    return W   


weights = gradient_descent(W)
# -----------------------------------------------------------------------------------------------------
# TESTING WEIGHTS
X_list = []
Y_list = []
for participant in testing_set:
    #get some info
    data_vectors = participant[1]
    number_times_participated = len(data_vectors)
    last_time_participated = 0
    age_lasttime = 0
    runtimes = []
    x = []
    
    for data in data_vectors:
        if int(data[-1]) > last_time_participated:
            last_time_participated = int(data[-1])
            age_lasttime = int(data[0])
        runtimes.append(data[2])
    
    #generate some info
    years_since_participated = current_year - last_time_participated
    age = age_lasttime + years_since_participated 
    #note, you might want to use age^2. Since as age increases, you tend be less likely to do physical activity in an expontential manner.
    avg_runtime = get_avgRuntime_inSec(runtimes)
    #print age, number_times_participated, years_since_participated, avg_runtime
    
    x.append(age)
    x.append(number_times_participated)
    x.append(years_since_participated)
    #x.append(avg_runtime)
    
    X_list.append(x)
    Y_list.append(participant[-1])

#X=np.array([np.array(xi) for xi in X_list])

#wTx = np.dot(X,np.transpose(W))


thresholds = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1, 0.11, 0.12, 0.13,0.14,0.15,0.2,0.3,0.4,0.5,0.6,0.7]
for t in thresholds:
    out = []
    for j in range(0,len(X_list)):
        x = X_list[j]
        calc = 0
        for i in range(0,len(x)):
            calc += x[i]*weights[i]
        calc = 1/(1 + np.exp(-calc))
        #print calc
        if calc >= t:
            out.append(1)
        else:
            out.append(0)
    
    
    FP = 0
    FN = 0
    TP = 0
    TN = 0
    
    for i in range(0,len(out)):
        if out[i] == Y_list[i] and out[i] == 1:
            TP += 1
        elif out[i] == Y_list[i] and out[i] == 0:
            TN += 1
        elif out[i] != Y_list[i] and out[i] == 1:
            FP +=1
        else:
            FN +=1
    
    #print "\nThreshold:\t%s"%t
    #print "Accuracy:", (TP + TN)/(0.0 + TP+TN+FP+FN)
    
    plt.plot(cost_values)
    plt.show()
    
    #for c in cost_values:
        #print c
        
    #print FP, FN, TP, TN
    print "%s,%s,%s,%s"%(t, FP/(0.0+FP+TN), TP/(0.0+TP+FN), (TP + TN)/(0.0 + TP+TN+FP+FN))
# -----------------------------------------------------------------------------------------------------
# TEST ON SET INCLUDING 2016
f = open("../raw_data/Project1_data.csv")
text = f.read()
lines = text.split("\n")

relevant_indices = [2, 4, 5, 7] #excluding ID

indexOf = {}

participant_data = []
for i in range(0,len(lines)):
    info = lines[i].split(",")
    relevant_features = []
    for r in relevant_indices:
        relevant_features.append(info[r])
    ID = int(info[0])
    if i==0:
        participant_data.append([ID])
        participant_data[-1].append([relevant_features])
        number_participants = 1
        continue
    if participant_data[-1][0] != ID:
        participant_data.append([ID])
        participant_data[-1].append([relevant_features])
        number_participants += 1
    else: 
        participant_data[-1][1].append(relevant_features) 

#######

X_list = []
ID_list = []
for participant in participant_data: #switch this to testing set
    #get some info
    data_vectors = participant[1]
    number_times_participated = len(data_vectors)
    last_time_participated = 0
    age_lasttime = 0
    runtimes = []
    x = []
    
    for data in data_vectors:
        if int(data[-1]) > last_time_participated:
            last_time_participated = int(data[-1])
            age_lasttime = int(data[0])
        runtimes.append(data[2])
    
    #generate some info
    years_since_participated = current_year - last_time_participated
    age = age_lasttime + years_since_participated 
    #note, you might want to use age^2. Since as age increases, you tend be less likely to do physical activity in an expontential manner.
    avg_runtime = get_avgRuntime_inSec(runtimes)
    #print age, number_times_participated, years_since_participated, avg_runtime
    
    x.append(age)
    x.append(number_times_participated)
    x.append(years_since_participated)
    #x.append(avg_runtime)
    
    X_list.append(x)
    ID_list.append(participant[0])

for i in range(0,len(ID_list)):
    print X_list[i], ID_list[i]
#######

#for p in participant_data:
 #   print p 

out = []
ID = []
for j in range(0,len(X_list)):
    x = X_list[j]
    calc = 0
    for i in range(0,len(x)):
        calc += x[i]*weights[i]
    calc = 1/(1 + np.exp(-calc))
    #print calc
    if calc >= 0.5:
        out.append(1)
        ID.append(ID_list[j])
    else:
        out.append(0)
        ID.append(ID_list[j])
    
for i in range(0,len(ID)):
    print ID[i], out[j]
    