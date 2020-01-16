import os
import sys
import numpy as np

log_dir = 'sa_log'

emb_type = sys.argv[1]
task = sys.argv[2]
model = sys.argv[3]
emb_name = os.path.basename(sys.argv[4])

# print emb_type, emb_name, task

tasks = ['mr', 'subj', 'cr', 'mpqa', 'trec', 'sst']

# print task

if task in tasks[:-1]:

    best_valid = -1e10
    best_valid_std = 1e10
    best_test = -1e10
    best_test_std = 1e10

    for dropout in [0.7]:
        valids = []
        tests = []
        for cv in range(10):
            # file_path = log_dir + '/' + task + '_' + emb_type + '_' + emb_name + '_cv_' + str(cv) + '_model_' + model + '_dropout_' + str(dropout) + '_seed_1234.log'
            file_path = os.path.join(log_dir, "{task}_{emb_type}_{emb_name}".format(
                task=task, emb_type=emb_type, emb_name=emb_name),
                "{task}_{emb_type}_{emb_name}_cv_{cv}_model_{model}_dropout_{do}_seed_1234.log".format(
                task=task, emb_type=emb_type, emb_name=emb_name, cv=cv, model=model, do=dropout))
            print file_path
            assert os.path.isfile(file_path), 'Cannot find the file!!'
            ff = open(file_path, 'r')
            dat = [_.strip() for _ in ff]
            # print dat[-2]
            # print dat[-3]
            valid = 1.0 - float(dat[-3].strip().split(': ')[1])
            test = 1.0 - float(dat[-2].strip().split(': ')[1])
            # print valid
            # print test

            valids.append(valid)
            tests.append(test)

        valid_mean = np.mean(np.array(valids))
        valid_mean_std = np.std(np.array(valids))
        test_mean = np.mean(np.array(tests))
        test_mean_std = np.std(np.array(tests))
        if valid_mean > best_valid or (valid_mean == best_valid and valid_mean_std < best_valid_std):
            best_valid = valid_mean
            best_valid_std = valid_mean_std
            best_test = test_mean
            best_test_std = test_mean_std
        # print best_valid, best_valid_std, best_test, best_test_std
    # print best_valid, best_valid_std, best_test, best_test_std
    print valid_mean, valid_mean_std



elif task == tasks[-1]:

    best_valid = -1e10
    best_valid_std = 1e10
    best_test = -1e10
    best_test_std = 1e10

    for dropout in [0.1, 0.3, 0.5, 0.7]:
        valids = []
        tests = []
        for seed in [1234, 1235, 1236, 1237, 1238]:
            file_path = emb_type + '/' + task + '_' + emb_type + '_' + emb_name + '_cv_' + str(0) + '_model_' + model + '_dropout_' + str(dropout) + '_seed_' + str(seed) + '.log'
            print file_path
            assert os.path.isfile(file_path), 'Cannot find the file!!'
            ff = open(file_path, 'r')
            dat = [_.strip() for _ in ff]
            print dat[127]
            print dat[128]
            valid = 1.0 - float(dat[127].strip().split(': ')[1])
            test = 1.0 - float(dat[128].strip().split(': ')[1])
            print valid
            print test

            valids.append(valid)
            tests.append(test)

        valid_mean = np.mean(np.array(valids))
        valid_mean_std = np.std(np.array(valids))
        test_mean = np.mean(np.array(tests))
        test_mean_std = np.std(np.array(tests))
        if valid_mean > best_valid or (valid_mean == best_valid and valid_mean_std < best_valid_std):
            best_valid = valid_mean
            best_valid_std = valid_mean_std
            best_test = test_mean
            best_test_std = test_mean_std
        print best_valid, best_valid_std, best_test, best_test_std
    print best_valid, best_valid_std, best_test, best_test_std



else:
    print 'Error in task!!!'


