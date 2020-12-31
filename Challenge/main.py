import pandas as pd
import numpy as np

# This is a sample Python script.

users = pd.read_csv("data/users.csv", delimiter=';', dtype={'userID':'int', 'gender':'str', 'age':'int', 'profession':'int'}, names=['userID', 'gender', 'age', 'profession'])
movies = np.array(pd.read_csv("data/movies.csv", delimiter=';', dtype={'movieID':'int', 'year':'int', 'movie':'str'}, names=['movieID', 'year', 'movie']))
ratings = np.array(pd.read_csv("data/ratings.csv", delimiter=';', dtype={'userID':'int', 'movieID':'int', 'rating':'int'}, names=['userID', 'movieID', 'rating']))
predictions = pd.read_csv("data/predictions.csv")
submission = pd.read_csv("data/submission.csv")


if __name__ == '__main__':
    #TO DO - create a collaborative filtering function like in example
    utility_matrix = np.zeros((len(users), len(movies)))

    for i in ratings:
        utility_matrix[i[0]-1, i[1]-1] = i[2]

