# %%
import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import numpy
import string
import random
import string
from sklearn import linear_model

# %%
TQDM_ON = True
if TQDM_ON:
    from tqdm import tqdm

# %%
ORIGIN_DATA_GZIP_PATH = "./train.json.gz"
TRAINSET_DEFAULT_SAVE_PATH = "./train.json"
PAIR_HOUR_VALIDSET_FILE_DEFAULT_PATH = "./pairs_Hours_validset.csv"
PAIR_PLAYED_VALIDSET_FILE_DEFAULT_PATH = "./pairs_Played_validset.csv"
PRED_HOUR_VALIDSET_FILE_DEFAULT_PATH = "./predictions_Hours_validset.csv"
PRED_PLAYED_VALIDSET_FILE_DEFAULT_PATH = "./predictions_Played_validset.csv"
PAIR_HOUR_TESTSET_FILE_DEFAULT_PATH = "./pairs_Hours.csv"
PAIR_PLAYED_TESTSET_FILE_DEFAULT_PATH = "./pairs_Played.csv"
PRED_HOUR_TESTSET_FILE_DEFAULT_PATH = "./predictions_Hours.csv"
PRED_PLAYED_TESTSET_FILE_DEFAULT_PATH = "./predictions_Played.csv"
q5_output_path="HWpredictions_Played.csv"
q8_output_path="HWpredictions_Hours.csv"

# %%
random.seed(0)

# %%
def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N

# %%
def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)
def readJSON(path):
    f = gzip.open(path, 'rt',encoding="utf-8")
    f.readline()
    for l in f:
        d = eval(l)
        u = d['userID']
        g = d['gameID']
        yield u,g,d

# %%
def read_raw_json_file(path:str):
    with open(path,encoding="utf-8") as fin:
        lines = fin.readlines()
    for l in lines:
        d = eval(l)
        u = d['userID']
        g = d['gameID']
        yield u,g,d

# %%
answers = {}
# Some data structures that will be useful
allHours = []
for l in readJSON("./train.json.gz"):
    allHours.append(l)
hoursTrain = allHours[:165000]
hoursValid = allHours[165000:]

# %%
# record wich games every user has played
games_per_user = defaultdict(set)
for uid,game_id,_ in allHours:
    games_per_user[uid].add(game_id)

# %%
all_games = set()
for _,game_id,_ in hoursTrain:
    all_games.add(game_id)
all_games = list(all_games)

# %%
valid_set_with_neg_sample = list()

for uid,game_id,_ in hoursValid:
    games_this_user_played = games_per_user[uid]
    rand_sample_game = all_games[random.randint(0,len(all_games)-1)]
    while rand_sample_game in games_this_user_played:
        rand_sample_game = all_games[random.randint(0,len(all_games)-1)]
    valid_set_with_neg_sample.append((uid,game_id,1))
    valid_set_with_neg_sample.append((uid,rand_sample_game,0))

# %%
def save_as_file_for_eval(dataset,path:str):
    with open(path,"w+",encoding="utf-8") as fout:
        fout.write("userID,gameID,prediction\n")
        for datum in dataset:
            uid = datum[0]
            game_id = datum[1]
            fout.write(f"{uid},{game_id}\n")

# %%
def save_train_set(train_set,path:str):
    assert type(train_set)==list
    t0 = train_set[0]
    assert type(t0)==tuple
    assert type(t0[2])==dict
    with open(path,"w+",encoding="utf-8") as fout:
        for datum in train_set:
            fout.write(str(datum[2])+"\n")

# %%
save_train_set(hoursTrain,TRAINSET_DEFAULT_SAVE_PATH)
save_as_file_for_eval(valid_set_with_neg_sample,PAIR_PLAYED_VALIDSET_FILE_DEFAULT_PATH)

# %%
def run_baseline_model(*,train_file_path:str=TRAINSET_DEFAULT_SAVE_PATH,
                       pair_hour_file_path:str=PAIR_HOUR_VALIDSET_FILE_DEFAULT_PATH,
                       pair_played_file_path:str=PAIR_PLAYED_VALIDSET_FILE_DEFAULT_PATH,
                       prediction_hour_output_path:str=PRED_HOUR_VALIDSET_FILE_DEFAULT_PATH,
                       prediction_played_output_path:str=PRED_PLAYED_VALIDSET_FILE_DEFAULT_PATH):
    allHours = []
    userHours = defaultdict(list)

    for user, game, d in read_raw_json_file(train_file_path):
        h = d["hours_transformed"]
        allHours.append(h)
        userHours[user].append(h)

    globalAverage = sum(allHours) / len(allHours)
    userAverage = {}
    for u in userHours:
        userAverage[u] = sum(userHours[u]) / len(userHours[u])

    predictions = open(prediction_hour_output_path, "w")
    for l in open(pair_hour_file_path):
        if l.startswith("userID"):
            # header
            predictions.write(l)
            continue
        u, g = l.strip().split(",")
        if u in userAverage:
            predictions.write(u + "," + g + "," + str(userAverage[u]) + "\n")
        else:
            predictions.write(u + "," + g + "," + str(globalAverage) + "\n")

    predictions.close()

    ### Would-play baseline: just rank which games are popular and which are not, and return '1' if a game is among the top-ranked

    gameCount = defaultdict(int)
    totalPlayed = 0

    for user, game, _ in read_raw_json_file(train_file_path):
        gameCount[game] += 1
        totalPlayed += 1

    mostPopular = [(gameCount[x], x) for x in gameCount]
    mostPopular.sort()
    mostPopular.reverse()

    return1 = set()
    count = 0
    for ic, i in mostPopular:
        count += ic
        return1.add(i)
        if count > totalPlayed / 2:
            break

    predictions = open(prediction_played_output_path, "w")
    for l in open(pair_played_file_path):
        if l.startswith("userID"):
            # header
            predictions.write(l)
            continue
        u, g = l.strip().split(",")
        if g in return1:
            predictions.write(u + "," + g + ",1\n")
        else:
            predictions.write(u + "," + g + ",0\n")

    predictions.close()

# %%
run_baseline_model()

# %%
valid_set_with_neg_sample_map = {(u,g):res for u,g,res in valid_set_with_neg_sample}
def calculate_pred_play_accu(pred_played_file_path:str=PRED_PLAYED_VALIDSET_FILE_DEFAULT_PATH):
    correct_cnt,total_cnt=0,0
    with open(pred_played_file_path) as fin:
        lines = fin.readlines()
        assert lines[0].startswith("userID,gameID,prediction"),f"csv file {pred_played_file_path} has wrong header"
        for line in lines[1:]:
            uid,gid,pred_str = line.strip().split(",")
            total_cnt+=1
            correct_cnt+=int(pred_str)==valid_set_with_neg_sample_map[(uid,gid)]
    assert total_cnt==len(valid_set_with_neg_sample)
    return correct_cnt/total_cnt

accu1 = calculate_pred_play_accu()
answers['Q1'] = accu1

# %%
accu1

# %%
def my_pred_play_model2(threshold:float,
                       *,train_file_path:str=TRAINSET_DEFAULT_SAVE_PATH,
                       pair_played_file_path:str=PAIR_PLAYED_VALIDSET_FILE_DEFAULT_PATH,
                       prediction_played_output_path:str=PRED_PLAYED_VALIDSET_FILE_DEFAULT_PATH):
    
    gameCount = defaultdict(int)
    totalPlayed = 0

    for user, game, _ in read_raw_json_file(train_file_path):
        gameCount[game] += 1
        totalPlayed += 1

    mostPopular = [(gameCount[x], x) for x in gameCount]
    mostPopular.sort()
    mostPopular.reverse()

    return1 = set()
    count = 0
    for ic, i in mostPopular:
        count += ic
        return1.add(i)
        if count > totalPlayed * threshold:
            break

    predictions = open(prediction_played_output_path, "w")
    for l in open(pair_played_file_path):
        if l.startswith("userID"):
            # header
            predictions.write(l)
            continue
        u, g = l.strip().split(",")
        if g in return1:
            predictions.write(u + "," + g + ",1\n")
        else:
            predictions.write(u + "," + g + ",0\n")

    predictions.close()

# %%
def search_max(l,r,iter_times,each_split,cal_accu_func):
    iter_range = range(iter_times)
    if TQDM_ON:
        iter_range = tqdm(iter_range)
    for i in iter_range:
        thsld_with_accu = list()
        split_i_range = range(1,each_split)
        each_split_size = (r-l)/each_split
        if TQDM_ON:
            split_i_range = tqdm(split_i_range)
        for s_i in split_i_range:
            x_s_i = l+each_split_size*s_i
            # my_pred_play_model2(x_s_i)
            # accu_this = calculate_pred_play_accu()
            accu_this = cal_accu_func(x_s_i)
            thsld_with_accu.append((x_s_i,accu_this))
            thsld_with_accu.sort(key=lambda tup:tup[1],reverse=True)
            ths_max_accu = thsld_with_accu[0][0]
            max_accu = thsld_with_accu[0][1]
            l,r = ths_max_accu-each_split_size,ths_max_accu+each_split_size
    return (ths_max_accu,max_accu)





# %%
def cal_accu_func2(thsld):
    my_pred_play_model2(thsld)
    return calculate_pred_play_accu()

ans2 = search_max(0.0,1.0,5,6,cal_accu_func2)

# %%
# l,r=0.0,1.0
# iter_times = 5
# each_split = 6
# iter_range = range(iter_times)
# if TQDM_ON:
#     iter_range = tqdm(iter_range)
# for i in iter_range:
#     thsld_with_accu = list()
#     split_i_range = range(1,each_split)
#     each_split_size = (r-l)/each_split
#     if TQDM_ON:
#         split_i_range = tqdm(split_i_range)
#     for s_i in split_i_range:
#         threshold = l+each_split_size*s_i
#         my_pred_play_model2(threshold)
#         accu_this = calculate_pred_play_accu()
#         thsld_with_accu.append((threshold,accu_this))
#         thsld_with_accu.sort(key=lambda tup:tup[1],reverse=True)
#         ths_max_accu = thsld_with_accu[0][0]
#         max_accu = thsld_with_accu[0][1]
#         l,r = ths_max_accu-each_split_size,ths_max_accu+each_split_size
# answers["Q2"]=[ths_max_accu,max_accu]

# %%
answers["Q2"]=list(ans2)

# %%
def jaccard_sim(set1,set2):
    return len(set1.intersection(set2))/len(set1.union(set2))

# %%
def my_pred_play_model3(threshold:float,
                       *,train_file_path:str=TRAINSET_DEFAULT_SAVE_PATH,
                       pair_played_file_path:str=PAIR_PLAYED_VALIDSET_FILE_DEFAULT_PATH,
                       prediction_played_output_path:str=PRED_PLAYED_VALIDSET_FILE_DEFAULT_PATH):

    user_per_game = defaultdict(set)
    game_per_user = defaultdict(set)

    for user, game, _ in read_raw_json_file(train_file_path):
        user_per_game[game].add(user)
        game_per_user[user].add(game)



    predictions = open(prediction_played_output_path, "w")
    for l in open(pair_played_file_path):
        if l.startswith("userID"):
            # header
            predictions.write(l)
            continue
        u, g = l.strip().split(",")
        users_played_this_game = user_per_game[g]
        pred_res = 0
        for g2 in game_per_user[u]:
            if jaccard_sim(user_per_game[g2],users_played_this_game)>=threshold:
                pred_res=1
                break
        predictions.write(f"{u},{g},{pred_res}\n")

    predictions.close()

# %%
def cal_accu_func3(thsld):
    my_pred_play_model3(thsld)
    return calculate_pred_play_accu()

ans3 = search_max(0.0,1.0,6,6,cal_accu_func3)

# %%
ans3

# %%
answers["Q3"]=ans3[1]

# %%
gameCount = defaultdict(int)
totalPlayed = 0

user_per_game = defaultdict(set)
game_per_user = defaultdict(set)

for user, game, _ in read_raw_json_file(TRAINSET_DEFAULT_SAVE_PATH):
    gameCount[game] += 1
    totalPlayed += 1
    user_per_game[game].add(user)
    game_per_user[user].add(game)

mostPopular = [(gameCount[x], x) for x in gameCount]
mostPopular.sort()
mostPopular.reverse()

popular_games = set()
count = 0
for ic, i in mostPopular:
    count += ic
    popular_games.add(i)
    if count > totalPlayed * answers["Q2"][0]:
        break


def getX4(uid,game_id):
    users_played_this_game = user_per_game[game_id]
    max_jaccard_sim = 0.0
    game_user_played = game_per_user[uid]
    if len(game_user_played)>1:
        max_jaccard_sim = max(jaccard_sim(user_per_game[g2],users_played_this_game) for g2 in game_user_played if g2!=g)
    is_this_game_pop = game_id in popular_games
    return [float(is_this_game_pop),max_jaccard_sim]


X4 = list()
Y4 = list()
for u,g,_d in (hoursTrain if not TQDM_ON else tqdm(hoursTrain)):
    rand_sample_game = all_games[random.randint(0,len(all_games)-1)]
    while rand_sample_game in games_this_user_played:
        rand_sample_game = all_games[random.randint(0,len(all_games)-1)]
    x_pos = getX4(u,g)
    X4.append(x_pos)
    Y4.append(1.0)
    x_neg = getX4(u,rand_sample_game)
    X4.append(x_neg)
    Y4.append(0.0)


# %%
X4 = numpy.array(X4)
Y4 = numpy.array(Y4)
logistic_reg = linear_model.LogisticRegression()
logistic_reg.fit(X4,Y4)

# %%
tmp = list(valid_set_with_neg_sample_map.items())
X4_validset = [getX4(u,g) for ((u,g),_) in tmp]
Y4_validset_label = [label for (_,label) in tmp]

# %%
Y4_validset_pred = logistic_reg.predict(X4_validset)

# %%
correct_cnt = 0
for y4l,y4p in zip(Y4_validset_label,Y4_validset_pred):
    correct_cnt+=int((y4p>=0.5)==(y4l>=0.5))

# %%
accu4 = correct_cnt/len(Y4_validset_label)

# %%
answers["Q4"] = accu4

# %%
answers

# %%
with open(PAIR_PLAYED_TESTSET_FILE_DEFAULT_PATH) as fin:
    lines = fin.readlines()
    assert lines[0].startswith("userID,gameID,prediction")
    X5 = list()
    user_game_tuples = list()
    for line in lines[1:]:
        u,g=tuple(line.strip().split(","))
        user_game_tuples.append((u,g))
        X5.append(getX4(u,g))

# %%
X5 = numpy.array(X5)
Y5_pred = logistic_reg.predict(X5)
with open(q5_output_path,"w+") as fout:
    fout.write("userID,gameID,prediction\n")
    for (u,g),y_pred in zip(user_game_tuples,Y5_pred):
        y_pred_int = 1 if y_pred>=0.5 else 0
        fout.write(f"{u},{g},{y_pred_int}\n")
     

# %%
answers['Q5'] = "I confirm that I have uploaded an assignment submission to gradescope"

# %%
time_pred_train_valid_split=0.95 # 5% data used as valid set , 90% as train
split_pos = int(time_pred_train_valid_split*len(allHours))
time_pred_dataset_all = [(u,g,math.log2(1.0+d['hours'])) for (u,g,d) in allHours]
# random.shuffle(time_pred_dataset_all)
time_pred_trainset = time_pred_dataset_all[:split_pos]
time_pred_validset = time_pred_dataset_all[split_pos:]

# %%
nTrain = split_pos
regularization_lambda = 1.0

user_per_game = defaultdict(set)
game_per_user = defaultdict(set)
ug_time_mapping = dict()

for (u,g,transformed_t) in time_pred_trainset:
    user_per_game[g].add(u)
    game_per_user[u].add(g)
    ug_time_mapping[(u,g)]=transformed_t

hoursPerUser = {u:sum(ug_time_mapping[(u,g)] for g in game_per_user[u]) for u in game_per_user}
hoursPerItem = {g:sum(ug_time_mapping[(u,g)] for u in user_per_game[g]) for g in user_per_game}
globalAverage = sum(t for (_u,_g,t) in time_pred_trainset)/len(time_pred_trainset)

betaU = {}
betaI = {}
for u in hoursPerUser:
    betaU[u] = hoursPerUser[u]/len(game_per_user[u])-globalAverage

for g in hoursPerItem:
    betaI[g] = hoursPerItem[g]/len(user_per_game[g])-globalAverage

alpha = globalAverage 

assert len(betaU)==len(game_per_user)
assert len(betaI)==len(user_per_game)

def predict(u,g):
    res = alpha
    if u in betaU:
        res+=betaU[u]
    if g in betaI:
        res+=betaI[g]
    return res

def mse(label,pred):
    assert len(label)==len(pred)
    return sum((y-ypred)**2 for y,ypred in zip(label,pred))/len(label)

def eval_mse():
    labels = list()
    pred = list()
    for u,g,t in time_pred_validset:
        labels.append(t)
        pred.append(predict(u,g))
    return mse(labels,pred)


# print(f"mse before start is {eval_mse()}")
iter_times = 100
iter_times_range = range(iter_times)
if TQDM_ON:
    iter_times_range = tqdm(iter_times_range)
mse_valid = 1000000000.0
mse_continue_rising = 0
for i in iter_times_range:
    alpha_sum = 0.0
    for (u,g,t) in time_pred_trainset:
        alpha_sum+=t-betaU[u]-betaI[g]
    alpha_next = alpha_sum/nTrain
    alpha=alpha_next

    betaU_next = dict()
    for u in game_per_user:
        games_this_user_played = game_per_user[u]
        rating_delta_sum = 0.0
        for g in games_this_user_played:
            rating_delta_sum+=ug_time_mapping[(u,g)]-alpha-betaI[g]
        betaU_next[u] = rating_delta_sum/(len(games_this_user_played)+regularization_lambda)
    assert len(betaU)==len(betaU_next)
    betaU = betaU_next

    betaI_next = dict()
    for g in user_per_game:
        users_played_this_game = user_per_game[g]
        rating_delta_sum = 0.0
        for u in users_played_this_game:
            rating_delta_sum+=ug_time_mapping[(u,g)]-alpha-betaU[u]
        betaI_next[g] = rating_delta_sum/(len(users_played_this_game)+regularization_lambda)
    assert len(betaI)==len(betaI_next)
    betaI = betaI_next

    
    mse_this = eval_mse()
    if mse_this>mse_valid:
        mse_continue_rising+=1
        if mse_continue_rising>=2:
            break
    else:
        mse_valid = mse_this
        mse_continue_rising =0




# %%
answers["Q6"] = eval_mse()

# %%
answers["Q6"]

# %%
betaUs = [(betaU[u], u) for u in betaU]
betaIs = [(betaI[i], i) for i in betaI]
betaUs.sort()
betaIs.sort()

print("Maximum betaU = " + str(betaUs[-1][1]) + ' (' + str(betaUs[-1][0]) + ')')
print("Maximum betaI = " + str(betaIs[-1][1]) + ' (' + str(betaIs[-1][0]) + ')')
print("Minimum betaU = " + str(betaUs[0][1]) + ' (' + str(betaUs[0][0]) + ')')
print("Minimum betaI = " + str(betaIs[0][1]) + ' (' + str(betaIs[0][0]) + ')')

# %%
answers['Q7'] = [betaUs[-1][0], betaUs[0][0], betaIs[-1][0], betaIs[0][0]]

# %%
def q8_fit(regularization_lambda):    
    betaU = {}
    betaI = {}
    for u in hoursPerUser:
        betaU[u] = hoursPerUser[u]/len(game_per_user[u])-globalAverage

    for g in hoursPerItem:
        betaI[g] = hoursPerItem[g]/len(user_per_game[g])-globalAverage

    alpha = globalAverage 

    assert len(betaU)==len(game_per_user)
    assert len(betaI)==len(user_per_game)

    def predict(u,g):
        res = alpha
        if u in betaU:
            res+=betaU[u]
        if g in betaI:
            res+=betaI[g]
        return res

    def mse(label,pred):
        assert len(label)==len(pred)
        return sum((y-ypred)**2 for y,ypred in zip(label,pred))/len(label)

    def eval_mse():
        labels = list()
        pred = list()
        for u,g,t in time_pred_validset:
            labels.append(t)
            pred.append(predict(u,g))
        return mse(labels,pred)


    # print(f"mse before start is {eval_mse()}")
    iter_times = 100
    iter_times_range = range(iter_times)
    if TQDM_ON:
        iter_times_range = tqdm(iter_times_range)
    mse_valid = 1000000000.0
    mse_continue_rising = 0
    for i in iter_times_range:
        alpha_sum = 0.0
        for (u,g,t) in time_pred_trainset:
            alpha_sum+=t-betaU[u]-betaI[g]
        alpha_next = alpha_sum/nTrain
        alpha=alpha_next

        betaU_next = dict()
        for u in game_per_user:
            games_this_user_played = game_per_user[u]
            rating_delta_sum = 0.0
            for g in games_this_user_played:
                rating_delta_sum+=ug_time_mapping[(u,g)]-alpha-betaI[g]
            betaU_next[u] = rating_delta_sum/(len(games_this_user_played)+regularization_lambda)
        assert len(betaU)==len(betaU_next)
        betaU = betaU_next

        betaI_next = dict()
        for g in user_per_game:
            users_played_this_game = user_per_game[g]
            rating_delta_sum = 0.0
            for u in users_played_this_game:
                rating_delta_sum+=ug_time_mapping[(u,g)]-alpha-betaU[u]
            betaI_next[g] = rating_delta_sum/(len(users_played_this_game)+regularization_lambda)
        assert len(betaI)==len(betaI_next)
        betaI = betaI_next

        
        mse_this = eval_mse()
        if mse_this>mse_valid:
            mse_continue_rising+=1
            if mse_continue_rising>=2:
                break
        else:
            mse_valid = mse_this
            mse_continue_rising =0

    return alpha,betaU,betaI,eval_mse()

# %%
best_alpha,best_betaU,best_betaI,best_lambda,min_mse = None,None,None,None,10000000.0
for i in tqdm(range(-10,11)):
    regular_lambda = (2**i)
    a,bu,bi,mse1 = q8_fit(regular_lambda)
    if mse1<min_mse:
        best_alpha,best_betaU,best_betaI,best_lambda,min_mse = a,bu,bi,regular_lambda,mse1

# %%
best_lambda

# %%
best_alpha,best_betaU,best_betaI,best_lambda,min_mse = None,None,None,None,10000000.0
for i in tqdm(range(20)):
    regular_lambda = 2+0.3*i
    a,bu,bi,mse1 = q8_fit(regular_lambda)
    if mse1<min_mse:
        best_alpha,best_betaU,best_betaI,best_lambda,min_mse = a,bu,bi,regular_lambda,mse1

# %%
best_lambda,min_mse

# %%
def predict(u,g):
    res = best_alpha
    if u in betaU:
        res+=best_betaU[u]
    if g in betaI:
        res+=best_betaI[g]
    return res

# %%
answers['Q8'] = [best_lambda,min_mse]

# %%
predictions = open(q8_output_path, 'w')
for l in open(PAIR_HOUR_TESTSET_FILE_DEFAULT_PATH):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,g = l.strip().split(',')
    
    # Logic...
    
    _ = predictions.write(u + ',' + g + ',' + str(predict(u,g)) + '\n')

predictions.close()

# %%
f = open("answers_hw3.txt", 'w')
f.write(str(answers) + '\n')
f.close()


