import json
import numpy as np

printFormat = 1
folder = "hyperParSearch/"

names = []
for i in range(1, 4):
    names.append(f"{folder}nn_cup{i}.json")
names = [f'{folder}grid_2_nn_cup{1}.json']
results = []
for i in names:
    randResults = []
    file = open(i, 'r')
    results.extend(json.load(file))
    file.close()

#   x['valErrDev']
#   x['valE']
top = sorted(results, key=lambda x: x['valE']*1.0 + x['valErrDev']*0 + x['lambdaRegularization'])[:10]

#val R^2: {'%.4f' % top[i]['valA']} +/- {'%.4f' % top[i]['valAccDev']} | \
#tr R^2: {'%.4f' % top[i]['trA']} +/- {'%.4f' % top[i]['trAccDev']} | \
if printFormat == 0:
    for i in range(len(top)):
        print(f"val MEE: {'%.4f' % top[i]['valE']} +/- {'%.4f' % top[i]['valErrDev']} | \
tr MEE: {'%.4f' % top[i]['trE']} +/- {'%.4f' % top[i]['trErrDev']} | \
preprocess: {top[i]['preprocess']} | \
lambda: {top[i]['lambdaRegularization']} | \
delta_max: {top[i]['delta_max']} | \
delta_0: {top[i]['delta_0']} | \
standardInit: {top[i]['standardInit']} | \
model: {top[i]['model']}\
")

elif printFormat == 1:
    for i in range(len(top)):
        print(f"v MEE: {'%.4f' % top[i]['valE']} +/- {'%.4f' % top[i]['valErrDev']} | \
t MEE: {'%.4f' % top[i]['trE']} +/- {'%.4f' % top[i]['trErrDev']} | \
preprocess: {top[i]['preprocess']} | \
lambda: {'%.5f' % top[i]['lambdaRegularization']} | \
patience: {top[i]['patience']} | \
delta_max: {'%.1f' % top[i]['delta_max']} | \
delta_0: {'%.4f' % top[i]['delta_0']} | \
init: {top[i]['standardInit']} | \
model: {top[i]['model']}\
")
        
else:
    for i in range(len(top)):
        print(f"v MEE: {'%.4f' % top[i]['valE']} +/- {'%.4f' % top[i]['valErrDev']} | \
v R^2: {'%.4f' % top[i]['valA']} +/- {'%.4f' % top[i]['valAccDev']} | \
t MEE: {'%.4f' % top[i]['trE']} +/- {'%.4f' % top[i]['trErrDev']} | \
t R^2: {'%.4f' % top[i]['trA']} +/- {'%.4f' % top[i]['trAccDev']} | \
preprocess: {top[i]['preprocess']} | \
lambda: {top[i]['lambdaRegularization']} | \
alpha: {top[i]['momentum']} | \
eta: {top[i]['learningRate']} | \
batch_size: {top[i]['batch_size']} | \
standardInit: {top[i]['standardInit']} | \
model: {top[i]['model']}\
")