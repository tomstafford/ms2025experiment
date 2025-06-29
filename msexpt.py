"""
Metascience 2025 reviewer experiment analysis


The code of the embedding model follows this code
https://github.com/snsf-data/snsf-grant-similarity/blob/main/notebooks/grant_similarity_transformer.ipynb 


Note from GO 2025-06-27
"Regarding the data security issue, these smaller models are just deposited on the Hugging Face Hub, but are not trained further and also the inference is not done at the Hugging Face servers, but on your own laptop. So how it works is that the code accesses the Hugging Face repository with the model and downloads the model first before it uses it locally. So, no data should be sent from your laptop to Hugging Face.
If you want to have a “bulletproof” implementation, you can also just download the model directly from Hugging Face and save it in your folder next to your code script (you need to download the files one by one: allenai/specter2_base at main), and then load the model not from Hugging Face but from your folder instead. This has also the advantage that you don’t have to download the model every single time you run your script and it also works 100% offline so you could basically turn off your internet connection to be 100% sure no data is transferred from your computer.
Here is an example how we have done this in another projects (lines 75 - 85): https://github.com/snsf-data/ml-peer-review-analysis/blob/2097ad3f24e095142bac884afc9b0907a2428a8a/code/binary_classification/binary_classification.py#L84"


"""

##### ---------------------------------------------------SET UP ENVIRONMENT #####


'''
conda activate msexpt

'''

# parameters

if False:
    #export environment in which this was last run
    os.system('conda env export > msexpt.yml') 


plottype = ''
plottype = '_TF-IDF'


# Libraries

import pandas as pd
import socket #to get host machine identity
import os #file and folder functions
import matplotlib.pyplot as plt #for making plots

# these required by snsf notebook code
import platform
import numpy as np

# import pytorch and transformers
import torch
from transformers import AutoTokenizer, AutoModel

# import similarity metrics from scikit-learn
from sklearn.metrics.pairwise import cosine_similarity
# and lanuage detection
from langdetect import detect

# for analysis
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

print("identifying host machine")
#test which machine we are on and set working directory
if 'tom' in socket.gethostname():
    os.chdir('/home/tom/t.stafford@sheffield.ac.uk/A_UNIVERSITY/expts/ms_expt')
else:
    print("Not sure whose machine we are on. Maybe the script will run")
              
##### ---------------------------------------------------- GET DATA ------- #####


# import reviewer consent
df_consent = pd.read_excel('data_raw/Reviewer consent.xlsx')

# import keywords
#df_keywords = pd.read_csv('data_raw/PC Keywords (Anon).xlsx')
# this didn't work so we save the FIRST SHEET only as tsv
df_keywords = pd.read_csv('data_conv/keywords.csv',sep='\t')

# do exlusions only
EXCLUDE_CORE = False

# TK this needs linking to reviewer codes to allow exclusions
df_consent[df_consent['Consent'] == 'Yes'] 
# missing line here to actually implement

if EXCLUDE_CORE:
    #currently this crashes the code, because later analysis tries to look up excluded reviewers?
    df_keywords = df_keywords[df_keywords['Core team'] != 'Yes']
    print("Exluding data from core team")

df_keywords.drop('Core team',axis=1,inplace=True)


# import submissions and review scorees
df = pd.read_excel('data_raw/Review.xlsx')

# exclude submissionis which were accepted/rejected without review
print("excluding {} unreviewed submission".format(sum(df['Summary score']>=7)))
df = df[~(df['Summary score']>=7)]



##### --------------------------------------------------- PREP DATA ------- #####

'''
TEST DATA
        'Submission RefNo', 'Title', 'Formats', 'Short Abstract',
       'Long Abstract', 'Reviewer comments (if any)', 'Reviewer1 ID',
       'Reveiwer1 Clarity', 'Reviewer1 Importance', 'Reviewer1 Novelty',
       'Reviewer1 Breadth', 'Reviewer1 Quality', 'Reviewer1 Suitability',
       'Reviewer2 ID', 'Reviewer2 Clarity', 'Reviewer2 Importance',
       'Reviewer2 Novelty', 'Reviewer2 Breadth', 'Reviewer2 Quality',
       'Reviewer2 Suitability', 'Reviewer3 ID', 'Reviewer3 Clarity',
       'Reviewer3 Importance', 'Reviewer3 Novelty', 'Reviewer3 Breadth',
       'Reviewer3 Quality', 'Reviewer3 Suitability'],

REAL DATA
        'PanelRefNo', 'Title', 'Formats', 'Short Abstract', 'Long Abstract',
       'Summary score', 'Average Quality', 'Average Breadth',
       'Average Clarity', 'Average Importance', 'Average Novelty',
       'Reviewer comments (if any)', 'Reviewer1 Name', 'Reveiwer1 Clarity',
       'Reviewer1 Importance', 'Reviewer1 Novelty', 'Reviewer1 Breadth',
       'Reviewer1 Quality', 'Reviewer1 Suitability', 'Reviewer2 Name',
       'Reviewer2 Clarity', 'Reviewer2 Importance', 'Reviewer2 Novelty',
       'Reviewer2 Breadth', 'Reviewer2 Quality', 'Reviewer2 Suitability',
       'Reviewer3 Name', 'Reviewer3 Clarity', 'Reviewer3 Importance',
       'Reviewer3 Novelty', 'Reviewer3 Breadth', 'Reviewer3 Quality',
       'Reviewer3 Suitability']
'''

suit_vars = ['Reviewer1 Suitability','Reviewer2 Suitability','Reviewer3 Suitability']

df['mean_suit']=df[suit_vars].mean(axis=1) #by paper

# Create separate DataFrames for each reviewer, including 'Submission RefNo'
reviewer1_df = df[['PanelRefNo', 'Reviewer1 Name', 'Reviewer1 Suitability']].rename(columns={
    'PanelRefNo': 'paperID',
    'Reviewer1 Name': 'reviewerID',
    'Reviewer1 Suitability': 'suitability'
})

reviewer2_df = df[['PanelRefNo', 'Reviewer2 Name', 'Reviewer2 Suitability']].rename(columns={
    'PanelRefNo': 'paperID',
    'Reviewer2 Name': 'reviewerID',
    'Reviewer2 Suitability': 'suitability'
})

reviewer3_df = df[['PanelRefNo', 'Reviewer3 Name', 'Reviewer3 Suitability']].rename(columns={
    'PanelRefNo': 'paperID',
    'Reviewer3 Name': 'reviewerID',
    'Reviewer3 Suitability': 'suitability'
})

# Concatenate all reviewer DataFrames
all_reviews_df = pd.concat([reviewer1_df, reviewer2_df, reviewer3_df])

all_reviews_df.reset_index(inplace=True)

all_reviews_df.drop('index',axis=1,inplace=True)

# generate averages by paper or reviewer
all_reviews_df.groupby('reviewerID')['suitability'].mean()
all_reviews_df.groupby('paperID')['suitability'].mean()


# describe sample: n reviewers, n applications, n reviews

print("We have {} papers".format(len(all_reviews_df['paperID'].unique()))) #463

print("We have {} reviewers".format(len(all_reviews_df['reviewerID'].unique()))) #26

print("We have {} reviews".format(len(all_reviews_df))) #1389

##### --------------------------------------------------- ANALYSIS ------- #####

# 1 charactertise suitability responses (boxplot?)
# - reviewer perspective
# paper perspective

# suitabvility from REVIEWER perspective

#mean of means
rmean=all_reviews_df.groupby('reviewerID')['suitability'].mean().mean()

#plot distrinbution
x = all_reviews_df.groupby('reviewerID')['suitability'].mean().values
plt.clf()
plt.hist(x)
plt.xlabel('Mean suitability by REVIEWER')
plt.ylabel('Frequency')
plt.xlim(0.5,7.5)
# add anotation to top left
plt.annotate('Mean = {:.2f}'.format(rmean),
             xy=(0.02, 0.88),
             xycoords='axes fraction')
plt.savefig('plots/1_suitability_by_REVIEWER.png',dpi=300,bbox_inches='tight')


# suitabvility from PAPER perspective

#mean of means
pmean =all_reviews_df.groupby('paperID')['suitability'].mean().mean()

#plot distrinbution
x = all_reviews_df.groupby('paperID')['suitability'].mean().values
plt.clf()
plt.hist(x,color='hotpink')
plt.xlabel('Mean suitability by PROPOSAL')
plt.ylabel('Frequency')
plt.xlim(0.5,7.5)
# add anotation to top left
plt.annotate('Mean = {:.2f}'.format(pmean),
             xy=(0.02, 0.88),
             xycoords='axes fraction')
plt.savefig('plots/1_suitability_by_proposal.png',dpi=300,bbox_inches='tight')

##### ----------------------------------------- CALCULATE SIMILARITIES ------- #####
# now matrching, follolwing the SNSF notebooks

# use GPU if available
current_os = platform.system()
# active device based on OS
if current_os == 'Darwin':
    # specify device as mps for Mac
    device = 'mps'
    print('MPS will be used as a device.')
else:
    # check if gpu is available, if yes use cuda, if not stick to cpu
    # install CUDA here:https://pytorch.org/get-started/locally/
    if torch.cuda.is_available():
        # must be 'cuda:0', not just 'cuda', see: https://github.com/deepset-ai/haystack/issues/3160
        device = torch.device('cuda:0')
        print('GPU', torch.cuda.get_device_name(0) ,'is available and will be used as a device.')
    else:
        device = torch.device('cpu')
        print('No GPU available, CPU will be used as a device instead.'
              + 'Be aware that the computation time increases significantly.')
        
# our data is all reviewer keywors and all papers

#SUBMISSIONS
submissions = df[['PanelRefNo','Title','Short Abstract','Long Abstract']]
#change column name
submissions.rename(columns={'PanelRefNo': 'ItemID'},inplace=True)
# concatenate titles and abstracts
submissions['TitleAbstract'] = submissions.Title + '. ' + submissions['Long Abstract']  
# Sensitivity tests - sometimes embeddingsd work better with shorter texts:
#submissions['TitleAbstract'] = submissions.Title + '. ' + submissions['Short Abstract']  
#submissions['TitleAbstract'] = submissions.Title
# results are similar, not improved with shorter inputs
# lower case
submissions['TitleAbstract'] = submissions.TitleAbstract.str.lower()


#REVIEWERS
reviewers = df_keywords

#concatenate columns
reviewers['All_Keywords'] = reviewers.drop('et', axis=1).fillna('').astype(str).agg(' '.join, axis=1)

#rename columns
reviewers.rename(columns={'et': 'ItemID'},inplace=True)
reviewers.rename(columns={'All_Keywords': 'TitleAbstract'},inplace=True)

# Set this to true to test code with a small subset of data
if False:
    df1 = submissions[['ItemID','TitleAbstract']][:4]
    df2 = reviewers[['ItemID','TitleAbstract']][1:3]
else:
    df1 = submissions[['ItemID','TitleAbstract']]
    df2 = reviewers[['ItemID','TitleAbstract']]

# join in one long list nb whether or paper or reviewer is in ItemID
data = pd.concat([df1,df2],axis=0)

# extract texts as a list
texts = data.TitleAbstract.tolist()

# specify the model name (SPECTER: BERT model pre-trained on scientific texts and augmented by a citation graph)
model_name = 'allenai/specter2_base'
# load the tokenizer and the model from HuggingFace and pass it to device
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

# storage for encoded texts
encoded_text = list(range(len(texts)))
# tokenize the texts
for text_idx in range(len(encoded_text)):
    # and send it to device
    encoded_text[text_idx] = tokenizer(texts[text_idx],
                                       max_length=512, # for BERT models the context window is limited to 512 tokens
                                       truncation=True, # truncate the texts exceeding 512 tokens
                                       padding='max_length', # pad shorter texts to 512 tokens
                                       return_tensors = "pt").to(device)
    

# extract the CLS token (first special token summarizing the sentence level embeddings in BERT models) - 22 mins
# storage for embeddings
embeddings = {}

import time

# --- Top of the code block ---
start_time = time.time()

# run the inputs through the model (sequentially to not overload CUDA memory in Jupyter)
for text_idx in range(len(encoded_text)):
    # first get the model output
    with torch.no_grad():
        output = model(**encoded_text[text_idx])
    # First element of model_output contains all token embeddings (last hidden state)
    token_embeddings = output[0]
    # extract the first out of 512 tokens, i.e. the so-called CLS token (torch dimension: [1,512,768])
    cls_token = token_embeddings[:,0,:]
    # normalize the CLS tokens with L2 norm to get similarity as dot product
    cls_token = torch.nn.functional.normalize(cls_token, p=2, dim=1)
    # pass back to CPU and convert to numpy array
    embeddings[data.ItemID.iloc[text_idx]] = cls_token.detach().to('cpu').numpy()[0]


# --- Tail of the code block ---
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Code block executed in: {elapsed_time:.4f} seconds") #284 = ~5 minutes

# and save as pandas dataframe
embeddings = pd.DataFrame(embeddings.items(), columns=['ItemID', 'TextEmbedding'])
# and explode the list of embeddings into separate columns
embeddings = pd.DataFrame(embeddings["TextEmbedding"].tolist(),
                          columns=list(range(token_embeddings.shape[2])),
                          index=embeddings["ItemID"].to_list())

##  this line doesn't work if submissions.ItemID contains non-unique values

# compute the similarity matrix among all grant texts
text_similarity = pd.DataFrame(cosine_similarity(embeddings),
                               columns=data.ItemID).set_index(data.ItemID)


##### ----------------------------------------- CALCULATE SIMILARITIES ------- #####

'''
#this is the basic pattern
# search top 5 most similar grants (1st one is the grant itself)
top5_grants = text_similarity[grant_number].sort_values(ascending=False)[1:6].index.to_list()
'''

#make so propsal x reviwer list

#drop all rows which don't start with T in ItemID
text_similarity = text_similarity[text_similarity.index.str.startswith('T')]

#drop all columns which don't start with R in ItemID
columns = text_similarity.columns #all
columns = columns[columns.str.startswith('R')] #to keep
#only keep columns in columns
text_similarity = text_similarity[columns]



def get_best_reviewer(proposal,text_similarity):
    '''given proposal ID, return id and score of best matching reivewer'''
    # select the ItemID of the column with the highest score in that row
    # and return the score
    score = text_similarity.loc[proposal].max()
    # now get ID
    id = text_similarity.loc[proposal].idxmax()
    return id,score

    
def get_best_proposal(reviewer,text_similarity):
    '''given reviewer ID, return id and score of best matching proposal'''
    return text_similarity[reviewer].idxmax(),text_similarity[reviewer].max()

def get_score(proposal,reviewer,text_similarity):
    '''return score of particiular matchup'''
    return text_similarity[reviewer].loc[proposal]    

def get_proposal_reviewers(proposal,all_reviews_df):
    '''report who reviewed a proposal'''
    return all_reviews_df[all_reviews_df['paperID']==proposal]['reviewerID'].to_list()



text_similarity.index # check index is proposal ID

# score actual matches
# make df for resulrs
for proposal in text_similarity.index:
    reviewers = get_proposal_reviewers(proposal,all_reviews_df)
    for reviewer in reviewers:
        if not pd.isna(reviewer):
            score = get_score(proposal,reviewer,text_similarity)    
            #get index in all_reviews_df where paperID and reviewerID match
            index = all_reviews_df[(all_reviews_df['paperID']==proposal) & (all_reviews_df['reviewerID']==reviewer)].index
            # put score in all_reviews_df using index
            all_reviews_df.loc[index,'score'] = score
            

# plot distribution and average of observed            
pmean = all_reviews_df.groupby('paperID')['score'].mean().mean()

#plot distrinbution
x = all_reviews_df.groupby('paperID')['score'].mean().values
plt.clf()
plt.hist(x,color='orange')
plt.xlabel('Mean match score by PAPER')
plt.ylabel('Frequency')
if plottype=='':
    plt.xlim(0.8,1)
# add anotation to top left
plt.annotate('Mean = {:.2f}'.format(pmean),
             horizontalalignment='right', 
             xy=(0.50, 0.88),
             xycoords='axes fraction')
# add vertical line at mean
plt.axvline(x=pmean,color='black',linestyle='--')
plt.savefig('plots/2'+plottype+'_matchscore_by_paper.png',dpi=300,bbox_inches='tight')


boot_n = 1000
# plot params
llim = 0.89
anox = 0.31

# score random match and 

av_match_store =[]

for i in range(boot_n):

    #create empty df to store match scores
    random_match = pd.DataFrame(index = text_similarity.index,columns=['av_match']) 

    for proposal in text_similarity.index:
        # from all reviewers, pick 3 at random, returning match score
        av = np.mean(np.random.choice(text_similarity.loc[proposal],3))
        random_match.loc[proposal,'av_match'] = av

    #add grand average to list
    av_match_store.append(random_match.mean().values[0])


#plot
#plot distrinbution
x = av_match_store
plt.clf()
plt.hist(x,color='blue',alpha=0.5)
plt.xlabel('Average mean match score by PAPER for random choice of reviewers,')
plt.ylabel('Frequency')
if plottype=='':
    plt.xlim(llim,0.97)
    plt.ylim(0,270)
# add anotation to top left
plt.annotate('Actual match\n = {:.2f}'.format(pmean),
             horizontalalignment='left', 
             xy=(anox, 0.78),
             xycoords='axes fraction')
plt.annotate('Random match\n = {:.2f}'.format(np.mean(av_match_store)), 
             color='blue',
             horizontalalignment='left', 
             xy=(anox, 0.88),
             xycoords='axes fraction')
# add vertical line at mean
plt.axvline(x=pmean,color='black',linestyle='--')
plt.savefig('plots/2'+plottype+'_matchscore_by_paper_boot.png',dpi=300,bbox_inches='tight')





# score best match

# NB ignores constraints of number of reviews each reviewer can do

#create empty df to store best match scores
best_match = pd.DataFrame(index = text_similarity.index,columns=['best_match'])

for proposal in text_similarity.index:
    best = text_similarity.loc[proposal].sort_values(ascending = False)[:3]
    best_match.loc[proposal,'best_match'] = np.mean(best)
    
bmean = np.mean(best_match)

np.mean(best_match)
#plot distrinbution
x = best_match
plt.clf()
plt.hist(x,color='turquoise',alpha=0.5)
plt.xlabel('Best possible match score by PAPER') 
plt.ylabel('Frequency')
if plottype=='':
    plt.xlim(llim,0.97)
    plt.ylim(0,270)
# add anotation to top left
plt.annotate('Actual match\n = {:.2f}'.format(pmean),
             horizontalalignment='left', 
             xy=(anox, 0.78),
             xycoords='axes fraction')
plt.annotate('Best match\n = {:.2f}'.format(bmean), 
             color='darkblue',
             horizontalalignment='left', 
             xy=(anox, 0.68),
             xycoords='axes fraction')
# add vertical line at mean
plt.axvline(x=pmean,color='black',linestyle='--')
plt.axvline(x=bmean,color='darkblue',linestyle='-.')
plt.savefig('plots/2'+plottype+'_matchscore_by_paper_best.png',dpi=300,bbox_inches='tight')


#combines
x = av_match_store
plt.clf()
plt.hist(x,color='blue',alpha=0.5)
x = best_match
#histogram with normalization
plt.hist(x,color='turquoise',alpha=0.5)
plt.xlabel('Average mean match score by PAPER')
plt.ylabel('Frequency')
plt.xlim(llim,0.97)
# add anotation to top left
plt.annotate('Actual match\n = {:.2f}'.format(pmean),
             horizontalalignment='left', 
             xy=(0.21, 0.78),
             xycoords='axes fraction')
plt.annotate('Random match\n = {:.2f}'.format(np.mean(av_match_store)), 
             color='blue',
             horizontalalignment='left', 
             xy=(0.21, 0.88),
             xycoords='axes fraction')
plt.annotate('Best match\n = {:.2f}'.format(bmean), 
             color='turquoise',
             horizontalalignment='left', 
             xy=(0.21, 0.68),
             xycoords='axes fraction')
# add vertical line at mean
plt.axvline(x=pmean,color='black',linestyle='--')
plt.savefig('plots/2_matchscore_by_paper_combined.png',dpi=300,bbox_inches='tight')

###3. within actual match correlation of suitability and matching

# calculate correlation


all_reviews_df[['suitability','score']].corr() 

# simple regression to predict suitability from score

#drop nans - ie rows where we have no match score (possibly because no reviewer ID)
all_reviews_df = all_reviews_df[~all_reviews_df['score'].isna()]
all_reviews_df = all_reviews_df[~all_reviews_df['suitability'].isna()]

#to be predicted
y = all_reviews_df['suitability'].values
# predictor
x = all_reviews_df['score'].values.reshape(-1,1)
# fit model


model = LinearRegression()
model.fit(x,y)
# predict
predicted = model.predict(x)
# plot
plt.clf()
plt.scatter(x,y,alpha=0.5)
plt.plot(x,predicted,color='red')
plt.xlabel('Match score')
plt.ylabel('Suitability')
plt.savefig('plots/3'+plottype+'_correlation_score-suitability.png',dpi=300,bbox_inches='tight')

#extract coefficient
coef = model.coef_
print(coef)

#extract intercept
intercept = model.intercept_
print(intercept)


# predict suitabilty = intercept + score * coef
def predict_suitability(intercept,coef,score):    
    '''use regression parameters to predict suitability from score'''
    return intercept + score * coef


pred_match = pd.DataFrame(index = text_similarity.index,columns=['actual_mean_suitability','best_mean_suitability'])

for proposal in text_similarity.index:
    #actual reviewers
    reviewers = get_proposal_reviewers(proposal,all_reviews_df)

    suit_list =[]
    for reviewer in reviewers:
        if not pd.isna(reviewer):
            mask = (all_reviews_df['paperID']==proposal) & (all_reviews_df['reviewerID']==reviewer)
            suitability = all_reviews_df[mask]['suitability'].values[0] 
            suit_list.append(suitability)
    pred_match.loc[proposal,'actual_mean_suitability'] = np.mean(suit_list)

    # best reviewers
    best = text_similarity.loc[proposal].sort_values(ascending = False)[:3]
    for ItemID in best.index:
        suit_list =[]
        score = best.loc[ItemID]
        suit_list.append(predict_suitability(intercept,coef,score)) # add predicted suitability from score
    pred_match.loc[proposal,'best_mean_suitability'] = np.mean(suit_list)

#plot
nbins=10
plt.clf()
x1=pred_match['actual_mean_suitability']
x2=pred_match['best_mean_suitability']
plt.annotate('Actual mean = {:.2f}'.format(np.mean(x1)),
             color='blue',
             horizontalalignment='left', 
             xy=(0.24, 0.935),
             xycoords='axes fraction')
plt.annotate('Best mean = {:.2f}'.format(np.mean(x2)), 
             color='red',
             horizontalalignment='left', 
             xy=(0.240, 0.88),             
             xycoords='axes fraction')


plt.hist([x1,x2],bins=10,color=['blue','red'],alpha=0.5)    
if plottype=='':
    plt.xlim(0.8,7.2)
plt.xlabel('Mean suitability')
plt.ylabel('Frequency')
plt.legend(['Actual','Best'])
plt.title('Actual and Predicted Best reviewer suitability ratings')
plt.savefig('plots/4'+plottype+'_correlation_score-suitability_bar.png',dpi=300,bbox_inches='tight')    

#import library for calculating Cohen's D
def cohens_d(group1, group2):
    # Calculating means of the two groups
    mean1, mean2 = np.mean(group1), np.mean(group2)
     
    # Calculating pooled standard deviation
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))
     
    # Calculating Cohen's d
    d = (mean1 - mean2) / pooled_std
     
    return d

#calculate cohen's D between x1 (actual) and x2 (best)
cohen_d = cohens_d(x2,x1)
print('Cohen\'s D = {:.2f}'.format(cohen_d))

# calculate kappa between x1 and x2

# Cohen'd 0.56 on Title+Full


# ----------------------------------Now replicate results for TF-IDF -------

# again , we are copying the procedure shared by SNSF
# 

# import NLP/text libraries
import nltk
import string

# import tfidf vectorizer and similarity metrics from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# download / update stopwords dictionary
nltk.download('stopwords')

# we use the same input data as above
type(texts)

# remove punctuation (string.punctuation: '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
for text_idx in range(len(texts)):
   texts[text_idx] = texts[text_idx].translate(str.maketrans('', '', string.punctuation))

   # create tokens as unigrams while removing stop words, use nltk english stopwords list
# create empty list for storage
tokens_unigrams = list()
# and loop through all texts
for text_idx in range(len(texts)):
   tokens_unigrams.append([word for word in texts[text_idx].split() if word not in nltk.corpus.stopwords.words('english')])
# perform stemming on unigrams
# load the Porter stemmer: https://www.nltk.org/api/nltk.stem.porter.html
ps = nltk.stem.PorterStemmer()
# and loop through all texts
for token_idx in range(len(tokens_unigrams)):
   # use again list comprehension to perform stemming word by word
   tokens_unigrams[token_idx] = [ps.stem(word) for word in tokens_unigrams[token_idx]]
   # keep only unigrams that have at least 2 characters, as otherwise the unigrams are not informative
   tokens_unigrams[token_idx] = [word for word in tokens_unigrams[token_idx] if len(word) > 1 ]

   # tokenize the text into n-grams as well, n is a tuning parameter (do not remove stop words here)
n_grams = 3
# only if n-grams are desired
if n_grams > 1:
   # create empty list for storage
   tokens_ngrams = list()
   # perform stemming first for all texts
   for text_idx in range(len(texts)):
      # stem words in text
      tokens_ngrams.append([ps.stem(word) for word in texts[text_idx].split()])
      # create n-grams from 2 up to n
      tokens_ngrams[text_idx] = nltk.everygrams(tokens_ngrams[text_idx], 2, n_grams)
      # and concatenate tuples and convert back to list of strings
      tokens_ngrams[text_idx] = [' '.join(token_idx) for token_idx in list(tokens_ngrams[text_idx])]
else:
   # otherwise return just an empty list
   tokens_ngrams = list()

# concatenate unigrams with n-grams to create the final vector of tokens
tokens = [(list_unigrams + list_ngrams) for list_unigrams, list_ngrams in zip(tokens_unigrams, tokens_ngrams)]

# compute tfidf via scikit-learn 
# initiate the vectorizer (identity function for tokenizer and preprocessor as we already tokenized the texts)
# specify l2 norm to get cosine similarity as dot product
tfidf = TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x, use_idf=True, norm='l2')  
# compute the tf-idf vector
tfidf_vector = tfidf.fit_transform(tokens)

# check the first element of the tf-idf vector for the tokens with largest weight
tfidf_df = pd.DataFrame(tfidf_vector[0].T.todense(), columns=["tf-idf"], index = tfidf.get_feature_names_out())
# sort the values
tfidf_df = tfidf_df.sort_values('tf-idf', ascending=False)
# check top 10
tfidf_df.head(10)

# compute the similarity matrix among all grant texts
text_similarity = pd.DataFrame(cosine_similarity(tfidf_vector),
                               columns=data.ItemID).set_index(data.ItemID)

# We can now repeat all our prior analysis but with TF-IDF