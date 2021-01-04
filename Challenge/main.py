import numpy as np
import pandas as pd
from random import randint

#####
##
## DATA IMPORT
##
#####

# Where data is located
movies_file = './data/movies.csv'
users_file = './data/users.csv'
ratings_file = './data/ratings.csv'
predictions_file = './data/predictions.csv'
submission_file = './data/submission.csv'

# Read the data using pandas
movies_description = pd.read_csv(movies_file, delimiter=';', dtype={'movieID': 'int', 'year': 'int', 'movie': 'str'},
                                 names=['movieID', 'year', 'movie'])
users_description = pd.read_csv(users_file, delimiter=';',
                                dtype={'userID': 'int', 'gender': 'str', 'age': 'int', 'profession': 'int'},
                                names=['userID', 'gender', 'age', 'profession'])
ratings_description = pd.read_csv(ratings_file, delimiter=';',
                                  dtype={'userID': 'int', 'movieID': 'int', 'rating': 'int'},
                                  names=['userID', 'movieID', 'rating'])
predictions_description = pd.read_csv(predictions_file, delimiter=';', names=['userID', 'movieID'])


#####
##
## COLLABORATIVE FILTERING
##
#####

def predict_collaborative_filtering(movies, users, ratings, predictions):
    # transform csv files to arrays
    movies = np.array(movies)
    users = np.array(users)
    ratings = np.array(ratings)

    # entries [0, :] and [:, 0] are empty to process data easier
    # users x movies matrix
    utility_matrix = np.zeros((len(users), len(movies)))

    # populate matrix with ratings
    for i in ratings:
        utility_matrix[i[0]-1, i[1]-1] = i[2]


    # calculate similarity matrix using pearson correlation coefficient

    # we first calculate the average movie rating per user
    averages = np.average(utility_matrix, axis=1, weights=(utility_matrix > 0))
    averages = averages[:, np.newaxis]


    #we normalize the ratings by subtracting the average if rating > 0
    normalized_matrix = np.where(np.array(utility_matrix > 0),  utility_matrix - averages, 0)

    print(utility_matrix[6038])
    print(averages[6038])
    print(normalized_matrix[6038])

    sim_matrix = np.zeros((len(normalized_matrix), len(normalized_matrix)))

    for i in range (0, len(normalized_matrix)):
        for m in range (i+1, len(normalized_matrix)):
            sim_matrix[i,m] = np.dot(normalized_matrix[i],normalized_matrix[m]) / (np.linalg.norm(normalized_matrix[i])
                                                                                   * np.linalg.norm(normalized_matrix[m]))

    print(sim_matrix)


    pass


#####
##
## LATENT FACTORS
##
#####

def predict_latent_factors(movies, users, ratings, predictions):
    ## TO COMPLETE

    pass


#####
##
## FINAL PREDICTORS
##
#####

def predict_final(movies, users, ratings, predictions):
    ## TO COMPLETE

    pass


def predict_randoms(movies, users, ratings, predictions):
    number_predictions = len(predictions)

    return [[idx, randint(1, 5)] for idx in range(1, number_predictions + 1)]


##########################################
# MAIN FUNCTION


### TESTING

predict_collaborative_filtering(movies_description, users_description, ratings_description, predictions_description)

#####
##
## SAVE RESULTS
##
#####


###################################
## commented out for later


# ## //!!\\ TO CHANGE by your prediction function
# predictions = predict_randoms(movies_description, users_description, ratings_description, predictions_description)
#
# # Save predictions, should be in the form 'list of tuples' or 'list of lists'
# with open(submission_file, 'w') as submission_writer:
#     # Formates data
#     predictions = [map(str, row) for row in predictions]
#     predictions = [','.join(row) for row in predictions]
#     predictions = 'Id,Rating\n' + '\n'.join(predictions)
#
#     # Writes it dowmn
#     submission_writer.write(predictions)
