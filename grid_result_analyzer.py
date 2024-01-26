import json
import numpy as np

name = 'grid_search_results_RProp.json'
name2 = 'grid_search_results_momentum.json'
name3 = 'random_search_results_RProp.json'

chosen = name3

file = open(chosen, 'r')
results = json.load(file)
file.close()

#   x['valErrDev']
#   x['valE']
top = sorted(results, key=lambda x: x['valE']*1.0 + x['valErrDev']*0.0)[:90]

if chosen == name:
    for i in range(len(top)):
        print(f"val MEE: {'%.4f' % top[i]['valE']} +/- {'%.4f' % top[i]['valErrDev']} | \
val R^2: {'%.4f' % top[i]['valA']} +/- {'%.4f' % top[i]['valAccDev']} | \
tr MEE: {'%.4f' % top[i]['trE']} +/- {'%.4f' % top[i]['trErrDev']} | \
tr R^2: {'%.4f' % top[i]['trA']} +/- {'%.4f' % top[i]['trAccDev']} | \
preprocess: {top[i]['preprocess']} | \
lambda: {top[i]['lambdaRegularization']} | \
delta_max: {top[i]['delta_max']} | \
delta_0: {top[i]['delta_0']} | \
standardInit: {top[i]['standardInit']} | \
model: {top[i]['model']}\
")

elif chosen == name3:
    for i in range(len(top)):
        print(f"v MEE: {'%.4f' % top[i]['valE']} +/- {'%.4f' % top[i]['valErrDev']} | \
v R^2: {'%.4f' % top[i]['valA']} +/- {'%.4f' % top[i]['valAccDev']} | \
t MEE: {'%.4f' % top[i]['trE']} +/- {'%.4f' % top[i]['trErrDev']} | \
t R^2: {'%.4f' % top[i]['trA']} +/- {'%.4f' % top[i]['trAccDev']} | \
preprocess: {top[i]['preprocess']} | \
lambda: {top[i]['lambdaRegularization']} | \
patience: {top[i]['patience']} | \
delta_max: {top[i]['delta_max']} | \
delta_0: {top[i]['delta_0']} | \
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