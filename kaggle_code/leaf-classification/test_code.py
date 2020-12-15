import numpy as np
import pandas as pd
from sklearn import manifold
from sklearn import svm
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense, Dropout, Input
from keras import regularizers
from subprocess import check_output
print(check_output(['ls', '../input']).decode('utf8'))
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
samplesub = pd.read_csv('../input/sample_submission.csv')
if 'samplesub' not in TANGSHAN:
    import csv
    if isinstance(samplesub, np.ndarray) or isinstance(samplesub, pd.DataFrame
        ) or isinstance(samplesub, pd.Series):
        shape_size = samplesub.shape
    elif isinstance(samplesub, list):
        shape_size = len(samplesub)
    else:
        shape_size = 'any'
    check_type = type(samplesub)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('samplesub')
        writer = csv.writer(f)
        writer.writerow(['samplesub', 25, check_type, shape_size])
x = np.float32(train.values[:, 2:])
x = np.concatenate((x, np.float32(test.values[:, 1:])), axis=0)
if 'test' not in TANGSHAN:
    import csv
    if isinstance(test, np.ndarray) or isinstance(test, pd.DataFrame
        ) or isinstance(test, pd.Series):
        shape_size = test.shape
    elif isinstance(test, list):
        shape_size = len(test)
    else:
        shape_size = 'any'
    check_type = type(test)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('test')
        writer = csv.writer(f)
        writer.writerow(['test', 28, check_type, shape_size])
x -= np.nanmin(x, axis=0)
if 'x' not in TANGSHAN:
    import csv
    if isinstance(x, np.ndarray) or isinstance(x, pd.DataFrame) or isinstance(x
        , pd.Series):
        shape_size = x.shape
    elif isinstance(x, list):
        shape_size = len(x)
    else:
        shape_size = 'any'
    check_type = type(x)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('x')
        writer = csv.writer(f)
        writer.writerow(['x', 30, check_type, shape_size])
x /= np.nanmax(x, axis=0)
labels = train.values[:, (1)]
if 'train' not in TANGSHAN:
    import csv
    if isinstance(train, np.ndarray) or isinstance(train, pd.DataFrame
        ) or isinstance(train, pd.Series):
        shape_size = train.shape
    elif isinstance(train, list):
        shape_size = len(train)
    else:
        shape_size = 'any'
    check_type = type(train)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('train')
        writer = csv.writer(f)
        writer.writerow(['train', 32, check_type, shape_size])
unique_labels = samplesub.columns.values[1:]
print('There are ', len(unique_labels), 'unique labels/species and ', len(x
    ), 'data points')
label_dict = dict(zip(unique_labels, range(len(unique_labels))))
nlabel_dict = dict(zip(range(len(unique_labels)), unique_labels))
if 'range' not in TANGSHAN:
    import csv
    if isinstance(range, np.ndarray) or isinstance(range, pd.DataFrame
        ) or isinstance(range, pd.Series):
        shape_size = range.shape
    elif isinstance(range, list):
        shape_size = len(range)
    else:
        shape_size = 'any'
    check_type = type(range)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('range')
        writer = csv.writer(f)
        writer.writerow(['range', 37, check_type, shape_size])
if 'unique_labels' not in TANGSHAN:
    import csv
    if isinstance(unique_labels, np.ndarray) or isinstance(unique_labels,
        pd.DataFrame) or isinstance(unique_labels, pd.Series):
        shape_size = unique_labels.shape
    elif isinstance(unique_labels, list):
        shape_size = len(unique_labels)
    else:
        shape_size = 'any'
    check_type = type(unique_labels)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('unique_labels')
        writer = csv.writer(f)
        writer.writerow(['unique_labels', 37, check_type, shape_size])
if 'nlabel_dict' not in TANGSHAN:
    import csv
    if isinstance(nlabel_dict, np.ndarray) or isinstance(nlabel_dict, pd.
        DataFrame) or isinstance(nlabel_dict, pd.Series):
        shape_size = nlabel_dict.shape
    elif isinstance(nlabel_dict, list):
        shape_size = len(nlabel_dict)
    else:
        shape_size = 'any'
    check_type = type(nlabel_dict)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('nlabel_dict')
        writer = csv.writer(f)
        writer.writerow(['nlabel_dict', 37, check_type, shape_size])
if 'dict' not in TANGSHAN:
    import csv
    if isinstance(dict, np.ndarray) or isinstance(dict, pd.DataFrame
        ) or isinstance(dict, pd.Series):
        shape_size = dict.shape
    elif isinstance(dict, list):
        shape_size = len(dict)
    else:
        shape_size = 'any'
    check_type = type(dict)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('dict')
        writer = csv.writer(f)
        writer.writerow(['dict', 37, check_type, shape_size])
nlabels = np.array([label_dict[l] for l in labels])
if 'l' not in TANGSHAN:
    import csv
    if isinstance(l, np.ndarray) or isinstance(l, pd.DataFrame) or isinstance(l
        , pd.Series):
        shape_size = l.shape
    elif isinstance(l, list):
        shape_size = len(l)
    else:
        shape_size = 'any'
    check_type = type(l)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('l')
        writer = csv.writer(f)
        writer.writerow(['l', 38, check_type, shape_size])
if 'label_dict' not in TANGSHAN:
    import csv
    if isinstance(label_dict, np.ndarray) or isinstance(label_dict, pd.
        DataFrame) or isinstance(label_dict, pd.Series):
        shape_size = label_dict.shape
    elif isinstance(label_dict, list):
        shape_size = len(label_dict)
    else:
        shape_size = 'any'
    check_type = type(label_dict)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('label_dict')
        writer = csv.writer(f)
        writer.writerow(['label_dict', 38, check_type, shape_size])
if 'labels' not in TANGSHAN:
    import csv
    if isinstance(labels, np.ndarray) or isinstance(labels, pd.DataFrame
        ) or isinstance(labels, pd.Series):
        shape_size = labels.shape
    elif isinstance(labels, list):
        shape_size = len(labels)
    else:
        shape_size = 'any'
    check_type = type(labels)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('labels')
        writer = csv.writer(f)
        writer.writerow(['labels', 38, check_type, shape_size])
if 'nlabels' not in TANGSHAN:
    import csv
    if isinstance(nlabels, np.ndarray) or isinstance(nlabels, pd.DataFrame
        ) or isinstance(nlabels, pd.Series):
        shape_size = nlabels.shape
    elif isinstance(nlabels, list):
        shape_size = len(nlabels)
    else:
        shape_size = 'any'
    check_type = type(nlabels)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('nlabels')
        writer = csv.writer(f)
        writer.writerow(['nlabels', 38, check_type, shape_size])
colors = ['r', 'g', 'b', 'm', 'k', 'c']
print('fitting isomap...')
ismp = manifold.Isomap(n_neighbors=10, n_components=2)
ismp.fit(x)
if 'ismp' not in TANGSHAN:
    import csv
    if isinstance(ismp, np.ndarray) or isinstance(ismp, pd.DataFrame
        ) or isinstance(ismp, pd.Series):
        shape_size = ismp.shape
    elif isinstance(ismp, list):
        shape_size = len(ismp)
    else:
        shape_size = 'any'
    check_type = type(ismp)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('ismp')
        writer = csv.writer(f)
        writer.writerow(['ismp', 46, check_type, shape_size])
plt.figure(1)
ax = plt.subplot(111)
y1 = ismp.transform(x)
plt.scatter(y1[990:, (0)], y1[990:, (1)], marker='.')
z = 0
for p1, p2 in zip(y1[:990, (0)], y1[:990, (1)]):
    plt.text(p1, p2, str(nlabels[z]), fontsize=8, color=colors[nlabels[z] % 6])
    if 'colors' not in TANGSHAN:
        import csv
        if isinstance(colors, np.ndarray) or isinstance(colors, pd.DataFrame
            ) or isinstance(colors, pd.Series):
            shape_size = colors.shape
        elif isinstance(colors, list):
            shape_size = len(colors)
        else:
            shape_size = 'any'
        check_type = type(colors)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('colors')
            writer = csv.writer(f)
            writer.writerow(['colors', 54, check_type, shape_size])
    if 'p1' not in TANGSHAN:
        import csv
        if isinstance(p1, np.ndarray) or isinstance(p1, pd.DataFrame
            ) or isinstance(p1, pd.Series):
            shape_size = p1.shape
        elif isinstance(p1, list):
            shape_size = len(p1)
        else:
            shape_size = 'any'
        check_type = type(p1)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('p1')
            writer = csv.writer(f)
            writer.writerow(['p1', 54, check_type, shape_size])
    z += 1
if 'p2' not in TANGSHAN:
    import csv
    if isinstance(p2, np.ndarray) or isinstance(p2, pd.DataFrame
        ) or isinstance(p2, pd.Series):
        shape_size = p2.shape
    elif isinstance(p2, list):
        shape_size = len(p2)
    else:
        shape_size = 'any'
    check_type = type(p2)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('p2')
        writer = csv.writer(f)
        writer.writerow(['p2', 53, check_type, shape_size])
plt.title('Isomap embedding')
plt.savefig('plot2.png')
plt.figure(2)
ax = plt.subplot(111)
if 'ax' not in TANGSHAN:
    import csv
    if isinstance(ax, np.ndarray) or isinstance(ax, pd.DataFrame
        ) or isinstance(ax, pd.Series):
        shape_size = ax.shape
    elif isinstance(ax, list):
        shape_size = len(ax)
    else:
        shape_size = 'any'
    check_type = type(ax)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('ax')
        writer = csv.writer(f)
        writer.writerow(['ax', 65, check_type, shape_size])
print('fitting t-SNE...')
tsne = manifold.TSNE(n_components=2, init='pca')
y2 = tsne.fit_transform(x)
if 'tsne' not in TANGSHAN:
    import csv
    if isinstance(tsne, np.ndarray) or isinstance(tsne, pd.DataFrame
        ) or isinstance(tsne, pd.Series):
        shape_size = tsne.shape
    elif isinstance(tsne, list):
        shape_size = len(tsne)
    else:
        shape_size = 'any'
    check_type = type(tsne)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('tsne')
        writer = csv.writer(f)
        writer.writerow(['tsne', 68, check_type, shape_size])
plt.scatter(y2[990:, (0)], y2[990:, (1)], marker='.')
z = 0
for p1, p2 in zip(y2[:990, (0)], y2[:990, (1)]):
    plt.text(p1, p2, str(nlabels[z]), fontsize=8, color=colors[nlabels[z] % 6])
    if 'str' not in TANGSHAN:
        import csv
        if isinstance(str, np.ndarray) or isinstance(str, pd.DataFrame
            ) or isinstance(str, pd.Series):
            shape_size = str.shape
        elif isinstance(str, list):
            shape_size = len(str)
        else:
            shape_size = 'any'
        check_type = type(str)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('str')
            writer = csv.writer(f)
            writer.writerow(['str', 72, check_type, shape_size])
    z += 1
if 'zip' not in TANGSHAN:
    import csv
    if isinstance(zip, np.ndarray) or isinstance(zip, pd.DataFrame
        ) or isinstance(zip, pd.Series):
        shape_size = zip.shape
    elif isinstance(zip, list):
        shape_size = len(zip)
    else:
        shape_size = 'any'
    check_type = type(zip)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('zip')
        writer = csv.writer(f)
        writer.writerow(['zip', 71, check_type, shape_size])
plt.title('t-SNE embedding')
plt.savefig('plot1.png')
print('fitting SVMs...')
svm_clf1 = svm.SVC(probability=True)
svm_clf1.fit(y1[:990, :], nlabels[:990])
svm_clf2 = svm.SVC(probability=True)
if 'svm_clf2' not in TANGSHAN:
    import csv
    if isinstance(svm_clf2, np.ndarray) or isinstance(svm_clf2, pd.DataFrame
        ) or isinstance(svm_clf2, pd.Series):
        shape_size = svm_clf2.shape
    elif isinstance(svm_clf2, list):
        shape_size = len(svm_clf2)
    else:
        shape_size = 'any'
    check_type = type(svm_clf2)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('svm_clf2')
        writer = csv.writer(f)
        writer.writerow(['svm_clf2', 87, check_type, shape_size])
svm_clf2.fit(y2[:990, :], nlabels[:990])
xx1, yy1 = np.meshgrid(np.arange(np.nanmin(y1[:, (0)]), np.nanmax(y1[:, (0)
    ]), 0.5), np.arange(np.nanmin(y1[:, (1)]), np.nanmax(y1[:, (1)]), 0.5))
if 'y1' not in TANGSHAN:
    import csv
    if isinstance(y1, np.ndarray) or isinstance(y1, pd.DataFrame
        ) or isinstance(y1, pd.Series):
        shape_size = y1.shape
    elif isinstance(y1, list):
        shape_size = len(y1)
    else:
        shape_size = 'any'
    check_type = type(y1)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('y1')
        writer = csv.writer(f)
        writer.writerow(['y1', 93, check_type, shape_size])
whole_space1 = np.concatenate((xx1.flatten()[:, (None)], yy1.flatten()[:, (
    None)]), axis=1)
if 'yy1' not in TANGSHAN:
    import csv
    if isinstance(yy1, np.ndarray) or isinstance(yy1, pd.DataFrame
        ) or isinstance(yy1, pd.Series):
        shape_size = yy1.shape
    elif isinstance(yy1, list):
        shape_size = len(yy1)
    else:
        shape_size = 'any'
    check_type = type(yy1)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('yy1')
        writer = csv.writer(f)
        writer.writerow(['yy1', 95, check_type, shape_size])
if 'xx1' not in TANGSHAN:
    import csv
    if isinstance(xx1, np.ndarray) or isinstance(xx1, pd.DataFrame
        ) or isinstance(xx1, pd.Series):
        shape_size = xx1.shape
    elif isinstance(xx1, list):
        shape_size = len(xx1)
    else:
        shape_size = 'any'
    check_type = type(xx1)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('xx1')
        writer = csv.writer(f)
        writer.writerow(['xx1', 95, check_type, shape_size])
xx2, yy2 = np.meshgrid(np.arange(np.nanmin(y2[:, (0)]), np.nanmax(y2[:, (0)
    ]), 0.5), np.arange(np.nanmin(y2[:, (1)]), np.nanmax(y2[:, (1)]), 0.5))
whole_space2 = np.concatenate((xx2.flatten()[:, (None)], yy2.flatten()[:, (
    None)]), axis=1)
plt.figure(4)
color = np.float32(svm_clf1.predict(whole_space1))
if 'whole_space1' not in TANGSHAN:
    import csv
    if isinstance(whole_space1, np.ndarray) or isinstance(whole_space1, pd.
        DataFrame) or isinstance(whole_space1, pd.Series):
        shape_size = whole_space1.shape
    elif isinstance(whole_space1, list):
        shape_size = len(whole_space1)
    else:
        shape_size = 'any'
    check_type = type(whole_space1)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('whole_space1')
        writer = csv.writer(f)
        writer.writerow(['whole_space1', 102, check_type, shape_size])
if 'svm_clf1' not in TANGSHAN:
    import csv
    if isinstance(svm_clf1, np.ndarray) or isinstance(svm_clf1, pd.DataFrame
        ) or isinstance(svm_clf1, pd.Series):
        shape_size = svm_clf1.shape
    elif isinstance(svm_clf1, list):
        shape_size = len(svm_clf1)
    else:
        shape_size = 'any'
    check_type = type(svm_clf1)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('svm_clf1')
        writer = csv.writer(f)
        writer.writerow(['svm_clf1', 102, check_type, shape_size])
plt.pcolormesh(xx1, yy1, color.reshape(xx1.shape), cmap=plt.cm.bone)
if 'color' not in TANGSHAN:
    import csv
    if isinstance(color, np.ndarray) or isinstance(color, pd.DataFrame
        ) or isinstance(color, pd.Series):
        shape_size = color.shape
    elif isinstance(color, list):
        shape_size = len(color)
    else:
        shape_size = 'any'
    check_type = type(color)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('color')
        writer = csv.writer(f)
        writer.writerow(['color', 103, check_type, shape_size])
plt.scatter(y1[990:, (0)], y1[990:, (1)], marker='.', label='unlabeled')
z = 0
for p1, p2 in zip(y1[:990, (0)], y1[:990, (1)]):
    plt.text(p1, p2, str(nlabels[z]), fontsize=8, color=colors[nlabels[z] % 6])
    if 'z' not in TANGSHAN:
        import csv
        if isinstance(z, np.ndarray) or isinstance(z, pd.DataFrame
            ) or isinstance(z, pd.Series):
            shape_size = z.shape
        elif isinstance(z, list):
            shape_size = len(z)
        else:
            shape_size = 'any'
        check_type = type(z)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('z')
            writer = csv.writer(f)
            writer.writerow(['z', 107, check_type, shape_size])
    z += 1
plt.legend(prop={'size': 6})
plt.xlim([np.nanmin(y1[:, (0)]), np.nanmax(y1[:, (0)])])
plt.ylim([np.nanmin(y1[:, (1)]), np.nanmax(y1[:, (1)])])
plt.title('RBF-SVM classification: ISOMAP embedding')
plt.savefig('separation_isomap.png')
plt.figure(5)
color = np.float32(svm_clf2.predict(whole_space2))
if 'whole_space2' not in TANGSHAN:
    import csv
    if isinstance(whole_space2, np.ndarray) or isinstance(whole_space2, pd.
        DataFrame) or isinstance(whole_space2, pd.Series):
        shape_size = whole_space2.shape
    elif isinstance(whole_space2, list):
        shape_size = len(whole_space2)
    else:
        shape_size = 'any'
    check_type = type(whole_space2)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('whole_space2')
        writer = csv.writer(f)
        writer.writerow(['whole_space2', 116, check_type, shape_size])
plt.pcolormesh(xx2, yy2, color.reshape(xx2.shape), cmap=plt.cm.bone)
if 'xx2' not in TANGSHAN:
    import csv
    if isinstance(xx2, np.ndarray) or isinstance(xx2, pd.DataFrame
        ) or isinstance(xx2, pd.Series):
        shape_size = xx2.shape
    elif isinstance(xx2, list):
        shape_size = len(xx2)
    else:
        shape_size = 'any'
    check_type = type(xx2)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('xx2')
        writer = csv.writer(f)
        writer.writerow(['xx2', 117, check_type, shape_size])
if 'yy2' not in TANGSHAN:
    import csv
    if isinstance(yy2, np.ndarray) or isinstance(yy2, pd.DataFrame
        ) or isinstance(yy2, pd.Series):
        shape_size = yy2.shape
    elif isinstance(yy2, list):
        shape_size = len(yy2)
    else:
        shape_size = 'any'
    check_type = type(yy2)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('yy2')
        writer = csv.writer(f)
        writer.writerow(['yy2', 117, check_type, shape_size])
plt.scatter(y2[990:, (0)], y2[990:, (1)], marker='.', label='unlabeled')
z = 0
for p1, p2 in zip(y2[:990, (0)], y2[:990, (1)]):
    plt.text(p1, p2, str(nlabels[z]), fontsize=8, color=colors[nlabels[z] % 6])
    z += 1
plt.legend(prop={'size': 6})
plt.xlim([np.nanmin(y2[:, (0)]), np.nanmax(y2[:, (0)])])
plt.ylim([np.nanmin(y2[:, (1)]), np.nanmax(y2[:, (1)])])
plt.title('RBF-SVM classification: t-SNE embedding')
plt.savefig('separation_tSNE.png')
epochs = 15
if 'epochs' not in TANGSHAN:
    import csv
    if isinstance(epochs, np.ndarray) or isinstance(epochs, pd.DataFrame
        ) or isinstance(epochs, pd.Series):
        shape_size = epochs.shape
    elif isinstance(epochs, list):
        shape_size = len(epochs)
    else:
        shape_size = 'any'
    check_type = type(epochs)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('epochs')
        writer = csv.writer(f)
        writer.writerow(['epochs', 132, check_type, shape_size])
x -= 0.5


def onehot(X):
    X_1hot = np.zeros((len(X), np.nanmax(X) + 1))
    for k in range(len(X)):
        X_1hot[k, X[k]] = 1
    return X_1hot


y1h = onehot(nlabels)
if 'y1h' not in TANGSHAN:
    import csv
    if isinstance(y1h, np.ndarray) or isinstance(y1h, pd.DataFrame
        ) or isinstance(y1h, pd.Series):
        shape_size = y1h.shape
    elif isinstance(y1h, list):
        shape_size = len(y1h)
    else:
        shape_size = 'any'
    check_type = type(y1h)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('y1h')
        writer = csv.writer(f)
        writer.writerow(['y1h', 142, check_type, shape_size])
inp = Input(shape=(x.shape[1],))
D1 = Dropout(0.01)(inp)
if 'inp' not in TANGSHAN:
    import csv
    if isinstance(inp, np.ndarray) or isinstance(inp, pd.DataFrame
        ) or isinstance(inp, pd.Series):
        shape_size = inp.shape
    elif isinstance(inp, list):
        shape_size = len(inp)
    else:
        shape_size = 'any'
    check_type = type(inp)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('inp')
        writer = csv.writer(f)
        writer.writerow(['inp', 145, check_type, shape_size])
L1 = Dense(1024, init='uniform', activation='tanh', activity_regularizer=
    regularizers.activity_l1(0.01))(D1)
if 'D1' not in TANGSHAN:
    import csv
    if isinstance(D1, np.ndarray) or isinstance(D1, pd.DataFrame
        ) or isinstance(D1, pd.Series):
        shape_size = D1.shape
    elif isinstance(D1, list):
        shape_size = len(D1)
    else:
        shape_size = 'any'
    check_type = type(D1)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('D1')
        writer = csv.writer(f)
        writer.writerow(['D1', 146, check_type, shape_size])
L2 = Dense(len(unique_labels), init='uniform', activation='softmax')(L1)
model1 = Model(inp, L2)
if 'L2' not in TANGSHAN:
    import csv
    if isinstance(L2, np.ndarray) or isinstance(L2, pd.DataFrame
        ) or isinstance(L2, pd.Series):
        shape_size = L2.shape
    elif isinstance(L2, list):
        shape_size = len(L2)
    else:
        shape_size = 'any'
    check_type = type(L2)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('L2')
        writer = csv.writer(f)
        writer.writerow(['L2', 149, check_type, shape_size])
repNN1 = Model(inp, L1)
if 'L1' not in TANGSHAN:
    import csv
    if isinstance(L1, np.ndarray) or isinstance(L1, pd.DataFrame
        ) or isinstance(L1, pd.Series):
        shape_size = L1.shape
    elif isinstance(L1, list):
        shape_size = len(L1)
    else:
        shape_size = 'any'
    check_type = type(L1)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('L1')
        writer = csv.writer(f)
        writer.writerow(['L1', 151, check_type, shape_size])
model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[
    'categorical_accuracy', 'binary_crossentropy'])
repNN1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[
    'categorical_accuracy', 'binary_crossentropy'])
weight_name = 'NN_sparse_rep_H1_1024.h5'
try:
    model1.load_weights(weight_name)
    print('Yay! found existing weights...')
    if 'print' not in TANGSHAN:
        import csv
        if isinstance(print, np.ndarray) or isinstance(print, pd.DataFrame
            ) or isinstance(print, pd.Series):
            shape_size = print.shape
        elif isinstance(print, list):
            shape_size = len(print)
        else:
            shape_size = 'any'
        check_type = type(print)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('print')
            writer = csv.writer(f)
            writer.writerow(['print', 159, check_type, shape_size])
except IOError:
    model1.fit(x[:990, :], y1h[:990, :], nb_epoch=epochs, batch_size=15)
    model1.save_weights(weight_name, overwrite=True)
    if 'model1' not in TANGSHAN:
        import csv
        if isinstance(model1, np.ndarray) or isinstance(model1, pd.DataFrame
            ) or isinstance(model1, pd.Series):
            shape_size = model1.shape
        elif isinstance(model1, list):
            shape_size = len(model1)
        else:
            shape_size = 'any'
        check_type = type(model1)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('model1')
            writer = csv.writer(f)
            writer.writerow(['model1', 162, check_type, shape_size])
    if 'weight_name' not in TANGSHAN:
        import csv
        if isinstance(weight_name, np.ndarray) or isinstance(weight_name,
            pd.DataFrame) or isinstance(weight_name, pd.Series):
            shape_size = weight_name.shape
        elif isinstance(weight_name, list):
            shape_size = len(weight_name)
        else:
            shape_size = 'any'
        check_type = type(weight_name)
        with open('tas.csv', 'a+') as f:
            TANGSHAN.append('weight_name')
            writer = csv.writer(f)
            writer.writerow(['weight_name', 162, check_type, shape_size])
if 'IOError' not in TANGSHAN:
    import csv
    if isinstance(IOError, np.ndarray) or isinstance(IOError, pd.DataFrame
        ) or isinstance(IOError, pd.Series):
        shape_size = IOError.shape
    elif isinstance(IOError, list):
        shape_size = len(IOError)
    else:
        shape_size = 'any'
    check_type = type(IOError)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('IOError')
        writer = csv.writer(f)
        writer.writerow(['IOError', 160, check_type, shape_size])
NN_rep1 = repNN1.predict(x)
if 'repNN1' not in TANGSHAN:
    import csv
    if isinstance(repNN1, np.ndarray) or isinstance(repNN1, pd.DataFrame
        ) or isinstance(repNN1, pd.Series):
        shape_size = repNN1.shape
    elif isinstance(repNN1, list):
        shape_size = len(repNN1)
    else:
        shape_size = 'any'
    check_type = type(repNN1)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('repNN1')
        writer = csv.writer(f)
        writer.writerow(['repNN1', 164, check_type, shape_size])
NN_rep1 = tsne.fit_transform(NN_rep1)
plt.figure(6)
svm_clf3 = svm.SVC(probability=True)
if 'svm_clf3' not in TANGSHAN:
    import csv
    if isinstance(svm_clf3, np.ndarray) or isinstance(svm_clf3, pd.DataFrame
        ) or isinstance(svm_clf3, pd.Series):
        shape_size = svm_clf3.shape
    elif isinstance(svm_clf3, list):
        shape_size = len(svm_clf3)
    else:
        shape_size = 'any'
    check_type = type(svm_clf3)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('svm_clf3')
        writer = csv.writer(f)
        writer.writerow(['svm_clf3', 174, check_type, shape_size])
svm_clf3.fit(NN_rep1[:990, :], nlabels[:990])
print('SVM scores t-SNE: ', svm_clf2.score(y2[:990, :], nlabels))
if 'y2' not in TANGSHAN:
    import csv
    if isinstance(y2, np.ndarray) or isinstance(y2, pd.DataFrame
        ) or isinstance(y2, pd.Series):
        shape_size = y2.shape
    elif isinstance(y2, list):
        shape_size = len(y2)
    else:
        shape_size = 'any'
    check_type = type(y2)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('y2')
        writer = csv.writer(f)
        writer.writerow(['y2', 176, check_type, shape_size])
print('SVM scores NN & t-SNE: ', svm_clf3.score(NN_rep1[:990, :], nlabels))
if 'NN_rep1' not in TANGSHAN:
    import csv
    if isinstance(NN_rep1, np.ndarray) or isinstance(NN_rep1, pd.DataFrame
        ) or isinstance(NN_rep1, pd.Series):
        shape_size = NN_rep1.shape
    elif isinstance(NN_rep1, list):
        shape_size = len(NN_rep1)
    else:
        shape_size = 'any'
    check_type = type(NN_rep1)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('NN_rep1')
        writer = csv.writer(f)
        writer.writerow(['NN_rep1', 177, check_type, shape_size])
xx3, yy3 = np.meshgrid(np.arange(np.nanmin(NN_rep1[:, (0)]), np.nanmax(
    NN_rep1[:, (0)]), 0.5), np.arange(np.nanmin(NN_rep1[:, (1)]), np.nanmax
    (NN_rep1[:, (1)]), 0.5))
whole_space3 = np.concatenate((xx3.flatten()[:, (None)], yy3.flatten()[:, (
    None)]), axis=1)
color = np.float32(svm_clf3.predict(whole_space3))
if 'whole_space3' not in TANGSHAN:
    import csv
    if isinstance(whole_space3, np.ndarray) or isinstance(whole_space3, pd.
        DataFrame) or isinstance(whole_space3, pd.Series):
        shape_size = whole_space3.shape
    elif isinstance(whole_space3, list):
        shape_size = len(whole_space3)
    else:
        shape_size = 'any'
    check_type = type(whole_space3)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('whole_space3')
        writer = csv.writer(f)
        writer.writerow(['whole_space3', 181, check_type, shape_size])
plt.pcolormesh(xx3, yy3, color.reshape(xx3.shape), cmap=plt.cm.bone)
if 'xx3' not in TANGSHAN:
    import csv
    if isinstance(xx3, np.ndarray) or isinstance(xx3, pd.DataFrame
        ) or isinstance(xx3, pd.Series):
        shape_size = xx3.shape
    elif isinstance(xx3, list):
        shape_size = len(xx3)
    else:
        shape_size = 'any'
    check_type = type(xx3)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('xx3')
        writer = csv.writer(f)
        writer.writerow(['xx3', 182, check_type, shape_size])
if 'yy3' not in TANGSHAN:
    import csv
    if isinstance(yy3, np.ndarray) or isinstance(yy3, pd.DataFrame
        ) or isinstance(yy3, pd.Series):
        shape_size = yy3.shape
    elif isinstance(yy3, list):
        shape_size = len(yy3)
    else:
        shape_size = 'any'
    check_type = type(yy3)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('yy3')
        writer = csv.writer(f)
        writer.writerow(['yy3', 182, check_type, shape_size])
plt.scatter(NN_rep1[990:, (0)], NN_rep1[990:, (1)], marker='.')
z = 0
for p1, p2 in zip(NN_rep1[:990, (0)], NN_rep1[:990, (1)]):
    plt.text(p1, p2, str(nlabels[z]), fontsize=8, color=colors[nlabels[z] % 6])
    z += 1
plt.xlim([np.nanmin(NN_rep1[:, (0)]), np.nanmax(NN_rep1[:, (0)])])
plt.ylim([np.nanmin(NN_rep1[:, (1)]), np.nanmax(NN_rep1[:, (1)])])
plt.title('t-SNE embedding of NN hidden layer activations (1024 units, sparse)'
    )
plt.savefig('tSNE_neural_network.png')
