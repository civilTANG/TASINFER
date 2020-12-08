import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
import pickle
from scipy.sparse import hstack
path ='../input/hubber-models/'

# ------------------------ loading data ---------------------

print('Loading data...')
with open(path + 'main_categories.pickle','rb') as handle:
    main_categories = pickle.load(handle)
main_categories = [x.replace('&','') for x in main_categories]

test = pd.read_csv('../input/mercari-price-suggestion-challenge/test.tsv',sep='\t'
    ,usecols=['test_id','name','item_description','shipping','category_name','brand_name'],
    index_col =['test_id'])
    
train = pd.read_csv(path + 'train.csv',usecols = ['name','item_description','brand_name'])
    
print('Done!')
print('*'*70)
# ----------------- split categories of test -------------------
    
print('Spliting categories...')
def text_split(text):
    try:
        return text.split("/")[0]
    except:
        return "No Label"
        
test['main_cat'] = test['category_name'].apply(lambda x: text_split(x))
test['main_cat'] = [x.replace('&','') for x in test['main_cat']]
test.drop(['category_name'],axis=1,inplace=True)
print('Done!')
print('*'*70)

# ------------------- taking care of null values-----------
print('Filling null values...')
test['item_description'].fillna(value='No description yet',inplace=True)
test['name'].fillna(value='missing',inplace=True)
test['shipping'].fillna(value=0,inplace=True)
test['brand_name'].fillna(value='missing',inplace=True)
print('Done!')
print('*'*70)

# -------------------- creating vocabulary -------------

print('Creating vocabulary...')
name_train = train.name
name_test = test.name
names = np.concatenate((name_train,name_test))

desc_train = train.item_description
desc_test = test.item_description
desc = np.concatenate((desc_train,desc_test))

count_vec_names = CountVectorizer()
counts_all_names =count_vec_names.fit_transform(names)

count_vec = CountVectorizer()
counts_all = count_vec.fit_transform(desc)

tf_idf_names = TfidfTransformer()
tf_idf_names.fit(counts_all_names)

tf_idf = TfidfTransformer()
tf_idf.fit(counts_all)

print('Done!')
print('*'*70)

# ------------------- encoding brands ---------------------
print("Encoding brands...")

brands = pd.concat((train.brand_name,test.brand_name))

brand_encoder = LabelEncoder()
brand_encoder.fit(brands.str.lower())

del train

print('Done!')
print('*'*70)
# ------------------------ time to train! -----------------

print('Training and predicting...')

final_pred=pd.Series()

for category in main_categories:
    print(category)
    
    train_data = pd.read_csv(path + str(category) +'.csv',usecols=['train_id','name','item_description',
                    'shipping','brand_name','price'],index_col = ['train_id'])
    
    test_data = test.loc[test['main_cat']==str(category),['name','item_description','shipping'
                    ,'brand_name']]
    
    brand_train = brand_encoder.transform(train_data.brand_name.str.lower())
    brand_test = brand_encoder.transform(test_data.brand_name.str.lower())
    
    names_train = train_data.name
    names_test = test_data['name']
    
    count_names_train = count_vec_names.transform(names_train)
    count_names_test = count_vec_names.transform(names_test)
    
    X_train_names = tf_idf_names.transform(count_names_train)
    X_test_names = tf_idf_names.transform(count_names_test)
    
    desc_train = train_data.item_description
    desc_test = test_data['item_description']
    
    count_train = count_vec.transform(desc_train)
    count_test = count_vec.transform(desc_test)
    
    X_train_desc = tf_idf.transform(count_train)
    X_test_desc = tf_idf.transform(count_test)
    
    # print(X_train_names.shape)
    # print(X_train_desc.shape)
    
    X_train = hstack((train_data.shipping[:,None], X_train_names,X_train_desc))
    X_test = hstack((test_data.shipping[:,None],X_test_names,X_test_desc))
    
    # print(X_train.shape)
    # print(X_test.shape)
    
    Y_train = train_data.price
    
    model = Ridge(alpha=1.7)
    model.fit(X_train,Y_train)
    
    pred = np.exp(model.predict(X_test))
    pred = pd.Series(pred,index = test_data.index)
    
    final_pred = final_pred.add(pred,fill_value=0)
    print('Done!')
    
final_pred.to_csv('final_sub.csv',header=['price'])

# # train = pd.read_csv(path + 'train.csv',usecols = ['item_description','price'])
# test = pd.read_csv('../input/mercari-price-suggestion-challenge/test.tsv',sep='\t'
#     ,usecols=['test_id','item_description',''],index_col =['test_id'])





# desc_train = train.item_description
# desc_test = test.item_description

# desc = np.concatenate((desc_train,desc_test))
# print(len(desc))

# count_vec = CountVectorizer()
# counts_all = count_vec.fit_transform(desc)

# counts_train = count_vec.transform(desc_train)
# counts_test = count_vec.transform(desc_test)



# tf_idf = TfidfTransformer()
# tf_idf.fit(counts_all)

# X_train = tf_idf.transform(counts_train)
# Y_train = train.price

# X_test = tf_idf.transform(counts_test)

# model = HuberRegressor()
# model.fit(X_train,Y_train)
# pred = np.exp(model.predict(X_test))
# pred = pd.Series(pred,index = test.index)
# pred.to_csv("sub.csv",header=['price'])