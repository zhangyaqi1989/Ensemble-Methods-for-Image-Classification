'''
Author: Yaqi Zhang
Date: 12/17/17
University of Wisconsin-Madison
'''
from os import listdir
from os.path import isfile, join
from random import sample
from resnet_ensemble import *

if __name__ == "__main__":
    print("Hello World")
    probs = []
    (x_train, y_train), (x_test, y_test) = load_data() # cifar-10
    y_train_old = y_train[:].reshape((y_train.shape[0], ))
    y_test_old = y_test[:] # save for error calculation
    (x_train, y_train), (x_test, y_test) = preprocess(x_train, y_train, x_test, y_test)
    n_trains = x_train.shape[0]
    indexes = list(range(n_trains))
    model_dir = "models"
    model_names = [join(model_dir, f) for f in listdir(model_dir) if isfile(join(model_dir, f))]
    print(model_names)

    n_trials = 10
    n_models = 7

    meta_epochs = 100
    # stack_loading_model(saved_model_files, meta_epochs, filename="resnet-stack.txt")
    # stack_loading_model_super(saved_model_files, meta_epochs, filename="resnet-stack.txt")
    for i in range(n_trials):
        saved_model_files = sample(model_names, n_models)
        stack_loading_model(saved_model_files, meta_epochs, filename="resnet-stack.txt")

    '''
    model_files = ["stack.%d.final.hdf5" % (i) for i in range(7)]
    n_picks = 3
    picked = []
    models = []

    rights = []
    for i, model_file in enumerate(model_files):
        model = load_model(model_file)
        models.append(model)
        prob = model.predict(x_train, verbose=1)
        pred = np.argmax(prob, axis=1)
        right = (pred - y_train_old) == 0
        rights.append(right)
        print("model %d: %f %%" % (i, sum(right)/float(n_trains)))

    for i in range(n_picks):
        score = -float("Inf")
        if len(indexes) == 0:
            break
        for j in range(len(model_files)):
            if j in picked:
                continue
            else:
                right = rights[j]
                sum_right = sum(right[indexes])
                print(sum_right/float(n_trains))
                if sum_right > score:
                    score = sum_right
                    best_cand = j
        print("score = %d" % (score))
        indexes = list(set(indexes) - set(np.where(rights[best_cand] == True)[0].tolist()))
        # picked.append(models[best_cand])
        picked.append(best_cand)

    # model = picked[0]
    print(picked)
    for i in picked:
        right = rights[i]
        sum_right = sum(right)
        print(sum_right/float(n_trains))

    # scores = model.evaluate(x_train, y_train, verbose = 1)
    # print('Test loss:', scores[0])
    # print('Test accuracy:', scores[1])
    '''

