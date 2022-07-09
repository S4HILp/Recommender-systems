import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords

ratings = pd.read_csv("ratings.csv")
tags = pd.read_csv("tags.csv")
movies = pd.read_csv('movies.csv')
links = pd.read_csv('links.csv')

movies_copy1 = movies.copy(deep=True)

ratings.drop(columns="timestamp",inplace=True) # dropping timestamp column.

tags.drop(columns="timestamp",inplace=True) # dropping timestamp column.

all_genres = movies["genres"].str.split("|")
temp = []
nrows = movies.shape[0]

for i in range(nrows):
  nGen = np.shape(all_genres.iloc[i])[0]
  for j in range(nGen):
    temp.append(all_genres.iloc[i][j])


genres = list(set(temp))

genres.remove("IMAX")
genres.remove("(no genres listed)")  

genres = np.asarray(genres).reshape(1,-1)  
genres = np.reshape(genres,-1)  
genres

movies_copy2 = movies.copy(deep=True)

movies = movies.apply(lambda x: x["genres"].split("|"), axis=1)

movies_copy2.drop(columns="genres",inplace=True)
movies.head()



mlb = MultiLabelBinarizer()

transformed = mlb.fit_transform(movies)
movies_enc = pd.DataFrame(data=transformed, columns=mlb.classes_)

movies = pd.concat([movies_copy2,movies_enc],axis=1)

movies.drop(columns=["IMAX","(no genres listed)"],inplace=True)

movies.head()

genre_count = []

for genre in genres:
  genre_count.append(movies[genre].sum())

# %%
# plt.style.use("seaborn")

# plt.figure(figsize=(15,15))

# y = np.arange(0,4500,500)
# ax = sns.barplot(y=genre_count,x=genres)
# ax.bar_label(ax.containers[0])

# plt.xticks(rotation=35)
# plt.show()

ratings_mov = pd.merge(ratings,movies,how='inner',on='movieId')
userMovieMat = pd.pivot_table(ratings_mov,index=['userId'])
userTable = pd.pivot_table(ratings_mov,index=['userId','movieId'])
userTable

userMat = pd.pivot_table(ratings_mov,index=['userId'],aggfunc=np.sum)
userMat = userMat.drop(columns=["rating","movieId"])
userMat

userRange = np.arange(1,611,1)

cosineSim = pd.DataFrame(cosine_similarity(userMat,userMat),columns = userRange).set_index(userRange)
cosineSim   # based on type(genre) of movie watched by an USER 

ratings.head()

meanRating = []

for genre in range(len(genres)):
  specificGen = movies_copy1["genres"].str.contains(genres[genre])
  mov = movies_copy1[specificGen]
  rating_ = mov.merge(ratings, on='movieId', how='inner')
  meanRating.append(round(rating_["rating"].mean(),2))

mean_ratings = pd.DataFrame({"genres":genres,
                             "Mean_rating":meanRating})

mean_ratings


# %%
meanRatingMovie = ratings.groupby('movieId')["rating"].mean()  # Getting the mean rating of a movie
numOfUsersRated = ratings.groupby("movieId")["userId"].count() # Counting number of users who watched a particular movie

meanRatingIncl = pd.merge(movies,meanRatingMovie, how="inner", on="movieId")
meanRatingIncl = pd.merge(meanRatingIncl,numOfUsersRated, how="inner", on="movieId")

meanRatingIncl.rename(columns={"rating":"meanRating","userId":"numUsers"}, inplace=True)
 
meanRatingIncl[["movieId",'title','meanRating','numUsers']].head()

# %%
"""
#### Plotting top 10 movies watched and rated by users.

"""

# %%
mRIfiltered = meanRatingIncl[meanRatingIncl["numUsers"]>10] # Including movies which have been watched by more than 10 users.
mRIfiltered = mRIfiltered.sort_values(by=['numUsers','meanRating'],ascending=False) # Sorting values by descending order by users and then rating

x = mRIfiltered["title"].head(10)
y1 = mRIfiltered["numUsers"].head(10)
y2 = mRIfiltered["meanRating"].head(10)

# plt.figure(figsize=(15,10))

# ax1 = sns.barplot(x=x,y=y1,palette='icefire')
# ax1.bar_label(ax1.containers[0])

# ax2 = ax1.twinx()
# sns.lineplot(x=x,y=y2,ax=ax2,color='r',marker='o')

# ax1.set_xticklabels(ax1.get_xticklabels(),rotation=90)
# plt.show()

# %%
"""
#### String checking tags field for searching movies (basic)

"""

# %%
tags.head()

# %%
# Searching for movies using tags

recomMovid = []

# search = "leonardo"

def searchInTags(search):
  recomMovid = []
  for mId, t in zip(tags["movieId"],tags["tag"]):
    if search.casefold() in t.casefold():
      recomMovid.append(mId)

  recomMovid = list(set(recomMovid))
  return recomMovid # Movie ID of all movies with "leonardo" in the tag field

# %%
"""
#### Removing stop words

"""

# %%
stop_words = stopwords.words('english')
tags['tag'] = tags['tag'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))


import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import random


noUserRated = ratings.groupby('movieId')['rating'].agg('count') # number of users who rated a specific movie
noMovRated = ratings.groupby('userId')['rating'].agg('count') # number of movies rated by each user

itemUserMat = pd.pivot_table(ratings_mov,values='rating',index='movieId',columns=['userId']) # item-user sparse matrix

itemUserMat = itemUserMat.loc[noUserRated[noUserRated > 15].index,:] # removing all movies rated by less than 15 users
itemUserMat = itemUserMat.loc[:,noMovRated[noMovRated > 35].index] # removing all users who have rated less than 25 movies

IUM = itemUserMat.fillna(0,inplace=True)
itemUserMat.head()

csrIUM = csr_matrix(itemUserMat)
itemUserMat = itemUserMat.reset_index()

model = NearestNeighbors(metric='cosine', n_neighbors=10)
model.fit(csrIUM)


def recommend(search,n=10):
    id1 = list(movies_copy1[movies['title'].str.casefold().str.contains(search)]['movieId']) # searching in title
    id2 = searchInTags(search)  # searching in tags
    id2_len = len(id2) 
    movId = list(set(id1 + id2)) 
    if (len(id1)==0 and id2_len!=0):
        mov_name = movies_copy1[movies_copy1['movieId'].isin(id2)]
        temp = pd.DataFrame(mov_name)
        temp = temp.astype({"movieId": float}, errors='raise') 
        temp = temp.merge(links,on='movieId')
        temp.drop(columns=['genres','imdbId'],inplace=True)
        return temp
    elif len(movId):        
        mId = itemUserMat[itemUserMat['movieId'].isin(movId)]
        io = random.randint(0,len(mId)-1)
        mId = mId.index[io]
        
        # using knn to find top 10 movies based on sparse matrix created above
        distances, indices = model.kneighbors(csrIUM[mId],n_neighbors=n+1)     
        
        # mapping id and cosine distances together
        ids = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),key=lambda x: x[1])[:0:-1]
        
        # recomMoviesandDist = []
        recomMovies = []
        
        # creating a dataframe of recommended movies
        for val in ids:
            ind = itemUserMat.iloc[val[0]]['movieId']
            idx = movies[movies['movieId'] == ind].index
            # recomMoviesandDist.append({'title':movies.iloc[idx]['title'].values[0],'Distance':val[1]})
            recomMovies.append({'title':movies.iloc[idx]['title'].values[0],
            'movieId':ind})
        # df = pd.DataFrame(recomMoviesandDist,index=range(1,n+1))
        mov = pd.DataFrame(recomMovies)
        mov = mov.merge(links,on='movieId')
        mov.drop(columns='imdbId',inplace=True)
        return mov

    else:
        return "Search again please"


def popular(n):
    X = mRIfiltered[['title','movieId']].head(n)
    pop = pd.DataFrame(X)
    pop = pop.merge(links,on='movieId')
    pop = pop.astype({"movieId": float}, errors='raise')
    pop.drop(columns = 'imdbId',inplace=True)
    return pop  # returning top n popular movies