import numpy as np
import pandas as pd
from random import randint

#####
## NetIDs
## nmouman
## nntasi
#####




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
## GET PREDICTED RATINGS
##
#####

def get_prediction(predictions, pred_matrix):
    # result matrix for submission
    result = np.zeros((len(predictions_description), 2), dtype=object)

    count = 0

    # populate result matrix with rounded predictions
    for row in predictions.itertuples():
        result[count, 0] = count + 1
        result[count, 1] = pred_matrix[row[1] - 1, row[2] - 1]
        count += 1

    return result


#####
##
## COLLABORATIVE FILTERING
##
#####


def predict_collaborative_item_based(movies, users, ratings, predictions):
    # users x movies matrix
    utility_matrix = np.zeros((len(users), len(movies)))

    # populate utility matrix with ratings
    for i in ratings.itertuples():
        utility_matrix[i[1] - 1, i[2] - 1] = i[3]

    # find movies with no rating
    no_rating_indices = np.where(~utility_matrix.any(axis=0))[0]

    # calculate mean for each movie
    mean_item_ratings = np.zeros(utility_matrix.shape[1])

    for i in range(0, utility_matrix.shape[1]):
        if not (i in no_rating_indices):
            mean_item_ratings[i] = np.average(utility_matrix[:, i], weights=(utility_matrix[:, i] > 0))

    mean_item_ratings = mean_item_ratings[np.newaxis, :]

    # we normalize the ratings by subtracting the average if rating > 0
    ratings_diff = np.where(np.array(utility_matrix > 0), utility_matrix - mean_item_ratings, 0)

    # calculate similarity matrix using pearson correlation coefficient (add epsilon to avoid division by 0)
    item_similarity = np.corrcoef(ratings_diff.T + 1e-9)

    # initialize prediction_matrix
    pred = np.zeros((len(users), len(movies)))

    # collaborative filtering using knn algorithm

    for j in range(ratings_diff.shape[1]):
        # find indices of 50 most similar items
        top_k_items = [np.argsort(item_similarity[:, j])[:-50 - 1:-1]]
        for i in range(ratings_diff.shape[0]):
            pred[i, j] = item_similarity[j, :][top_k_items].dot(ratings_diff[i, :][top_k_items].T)
            pred[i, j] /= np.sum(np.abs(item_similarity[j, :][top_k_items]))

    # add mean of item to prediction
    pred += mean_item_ratings

    return pred


def predict_collaborative_filtering(movies, users, ratings, predictions):
    # users x movies matrix
    utility_matrix = np.zeros((len(users), len(movies)))

    # populate utility matrix with ratins
    for i in ratings.itertuples():
        utility_matrix[i[1] - 1, i[2] - 1] = i[3]

    # we first calculate the average movie rating per user
    mean_user_ratings = np.average(utility_matrix, axis=1, weights=(utility_matrix > 0))[:, np.newaxis]

    # we normalize the ratings by subtracting the average if rating > 0
    ratings_diff = np.where(np.array(utility_matrix > 0), utility_matrix - mean_user_ratings, 0)

    user_similarity = np.zeros((len(users), len(users)))

    # calculate similarity matrix using pearson correlation coefficient (add epsilon to avoid division by 0)
    user_similarity = np.corrcoef(ratings_diff + 1e-9)

    # prediction_matrix
    pred = np.zeros((len(users), len(movies)))

    # collaborative filtering using knn algorithm
    total = ratings_diff.shape[0] * ratings_diff.shape[1]

    for i in range(ratings_diff.shape[0]):
        # find indices of 50 most similar users
        top_k_users = [np.argsort(user_similarity[:, i])[:-50 - 1:-1]]
        for j in range(ratings_diff.shape[1]):
            pred[i, j] = user_similarity[i, :][top_k_users].dot(ratings_diff[:, j][top_k_users])
            pred[i, j] /= np.sum(np.abs(user_similarity[i, :][top_k_users]))

    # add mean of user to predictions
    pred = mean_user_ratings + pred

    return pred


#####
##
## LATENT FACTORS
##
#####


def predict_latent_factors(movies, users, ratings, predictions):
    # users x movies matrix
    utility_matrix = np.zeros((len(users), len(movies)))

    # populate utility matrix with ratings
    for i in ratings.itertuples():
        utility_matrix[i[1] - 1, i[2] - 1] = i[3]

    m, n = utility_matrix.shape

    # set random P and Q with 10 factors
    P = np.random.rand(m, 10)
    Q = np.random.rand(10, n)

    # set gamma and lambda
    # lambda -> regularization factor
    # gamma -> learning step
    lamda = 0.01
    gamma = 0.001

    # stochastic gradient descent
    for epoch in range(100):
        print(epoch)
        for row in ratings.itertuples():
            # row[1] -> user index
            # row[2] -> movie index
            # row[3] -> rating (we exclude the 0 ratings)

            # calculate error
            eui = row[3] - np.dot(P[row[1] - 1, :], Q[:, row[2] - 1])
            # update P and Q
            P[row[1] - 1, :] = P[row[1] - 1, :] + gamma * 2 * (eui * Q[:, row[2] - 1] - lamda * P[row[1] - 1, :])
            Q[:, row[2] - 1] = Q[:, row[2] - 1] + gamma * 2 * (eui * P[row[1] - 1, :] - lamda * Q[:, row[2] - 1])

    # predicted ratings
    pred = P @ Q

    return pred


#####
##
## FINAL PREDICTORS
## LATENT FACTORS WITH BIASES
##
#####

def predict_latent_factor_biases(movies, users, ratings, predictions):
    # users x movies matrix
    utility_matrix = np.zeros((len(movies), len(users)))

    # populate utility matrix with ratings
    for i in ratings.itertuples():
        utility_matrix[i[2] - 1, i[1] - 1] = i[3]

    # calculate mean of ratings per user
    mean_users = np.average(utility_matrix, axis=0, weights=(utility_matrix > 0))

    # find indices of movies with no rating
    no_rating_indices = np.where(~utility_matrix.any(axis=1))[0]

    mean_movies = np.zeros(utility_matrix.shape[0])

    # calculate mean of ratings per movie
    for i in range(0, utility_matrix.shape[0]):
        if not (i in no_rating_indices):
            mean_movies[i] = np.average(utility_matrix[i, :], weights=(utility_matrix[i, :] > 0))

    # find average of all ratings
    mean_rating = np.average(utility_matrix, weights=(utility_matrix > 0))

    # calculate biases of users
    user_biases = mean_users - mean_rating

    # calculate biases of movies
    movie_biases = mean_movies - mean_rating

    m, n = utility_matrix.shape

    num_factors = 100

    # set random P and Q with num_factors
    P = np.random.normal(size=(m, num_factors), scale=1.0 / num_factors)
    Q = np.random.normal(size=(n, num_factors), scale=1.0 / num_factors)

    # set gamma and lambda
    # lambda -> regularization factor
    # gamma -> learning step
    lamda = 0.01
    gamma = 0.001

    # stochastic gradient descent
    for epoch in range(60):
        print(epoch)
        for row in ratings.itertuples():
            # row[1] -> user index
            # row[2] -> movie index
            # row[3] -> rating (we exclude the 0 ratings)

            # calculate error
            eui = row[3] - np.dot(P[row[2] - 1, :], Q[row[1] - 1, :]) - user_biases[row[1] - 1] - movie_biases[
                row[2] - 1] - mean_rating

            # update p, q and biases
            P[row[2] - 1, :] = P[row[2] - 1, :] + gamma * (2*eui * Q[row[1] - 1, :] - lamda * P[row[2] - 1, :])
            Q[row[1] - 1, :] = Q[row[1] - 1, :] + gamma * (2*eui * P[row[2] - 1, :] - lamda * Q[row[1] - 1, :])
            user_biases[row[1] - 1] += gamma * (2*eui - lamda * user_biases[row[1] - 1])
            movie_biases[row[2] - 1] += gamma * (2*eui - lamda * movie_biases[row[2] - 1])

    # predictions
    pred = P @ Q.T + movie_biases[:, np.newaxis] + user_biases[np.newaxis, :] + mean_rating

    return pred.T

# final predictor
def predict_latent_factors_item_based(pred_latent_factors, pred_item_based):

    return (pred_latent_factors + pred_item_based)/2


def predict_randoms(movies, users, ratings, predictions):
    number_predictions = len(predictions)

    return [[idx, randint(1, 5)] for idx in range(1, number_predictions + 1)]


#####
##
## SAVE RESULTS
##
#####

# predictions with latent factors
# predictions = get_prediction(predictions_description,
#                              predict_latent_factors(movies_description, users_description,
#                                                     ratings_description, predictions_description))

# predictions with latent factors + biases
# predictions = get_prediction(predictions_description,
#                              predict_latent_factor_biases(movies_description, users_description,
#                                                           ratings_description, predictions_description))

# predictions with item based collaborative filtering
# predictions = get_prediction(predictions_description,
#                              predict_collaborative_item_based(movies_description, users_description,
#                                                               ratings_description, predictions_description))

# predictions with user based collaborative filtering
# predictions = get_prediction(predictions_description,
#                             predict_collaborative_filtering(movies_description, users_description,
#                                                             ratings_description, predictions_description))

# predictions with random ratings
# predictions = predict_randoms(movies_description, users_description, ratings_description, predictions_description)

# predictions cf + lf
# predictions = get_prediction(predictions_description,
#                              predict_latent_factors_item_based(predict_latent_factors(movies_description, users_description,
#                                                      ratings_description, predictions_description),
#                             predict_collaborative_item_based(movies_description, users_description,
#                                                                ratings_description, predictions_description)))

# predictions cf + lf with biases
predictions = get_prediction(predictions_description,
                             predict_latent_factors_item_based(predict_latent_factor_biases(movies_description, users_description,
                                                     ratings_description, predictions_description),
                            predict_collaborative_item_based(movies_description, users_description,
                                                               ratings_description, predictions_description)))

# Save predictions, should be in the form 'list of tuples' or 'list of lists'
with open(submission_file, 'w') as submission_writer:
    # Formates data
    predictions = [map(str, row) for row in predictions]
    predictions = [','.join(row) for row in predictions]
    predictions = 'Id,Rating\n' + '\n'.join(predictions)

    # Writes it dowmn
    submission_writer.write(predictions)

print("end")
