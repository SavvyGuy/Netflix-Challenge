import numpy as np
import pandas as pd
from random import randint

#####
##
## DATA IMPORT
##
#####

# Where data is located
from sklearn.metrics import mean_squared_error

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
    # users x movies matrix
    utility_matrix = np.zeros((len(users), len(movies)))

    # populate utility matrix with ratins
    for i in ratings.itertuples():
        utility_matrix[i[1] - 1, i[2] - 1] = i[3]

    # calculate similarity matrix using pearson correlation coefficient

    # we first calculate the average movie rating per user
    mean_user_ratings = np.average(utility_matrix, axis=1, weights=(utility_matrix > 0))[:, np.newaxis]

    # we normalize the ratings by subtracting the average if rating > 0
    ratings_diff = np.where(np.array(utility_matrix > 0), utility_matrix - mean_user_ratings, 0)

    user_similarity = np.zeros((len(users), len(users)))

    # similarity matrix for users using cosine similarity

    user_similarity = ratings_diff.dot(ratings_diff.T) + 1e-9

    norms = np.array([np.sqrt(np.diagonal(user_similarity))])

    user_similarity = user_similarity / norms / norms.T

    # prediction_matrix
    pred = np.zeros((len(users), len(movies)))

    # collaborative filtering using knn algorithm

    total = ratings_diff.shape[0] * ratings_diff.shape[1]
    p = 0

    for i in range(ratings_diff.shape[0]):
        top_k_users = [np.argsort(user_similarity[:, i])[:-50 - 1:-1]]
        for j in range(ratings_diff.shape[1]):
            pred[i, j] = user_similarity[i, :][top_k_users].dot(ratings_diff[:, j][top_k_users])
            pred[i, j] /= np.sum(np.abs(user_similarity[i, :][top_k_users]))
            p += 1
        print('Progress: {:4.2f}%'.format(p / total * 100))
    pred = mean_user_ratings + pred

    # collaborative filtering without knn algorithm
    #pred = mean_user_ratings + user_similarity.dot(ratings_diff) / np.sum(np.abs(user_similarity), axis=1)[:,
    #                                                               np.newaxis]

    # # result matrix for submission
    result = np.zeros((len(predictions), 2), dtype=object)

    count = 0


    # populate result matrix with rounded predictions
    for row in predictions.itertuples():
        result[count, 0] = count + 1
        result[count, 1] = pred[row[1] - 1, row[2] - 1]
        count += 1


    return result


#####
##
## LATENT FACTORS
##
#####

# calculation for rmse
def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return np.sqrt(mean_squared_error(prediction, ground_truth))

def predict_latent_factors(movies, users, ratings, predictions):
    # # users x movies matrix
    utility_matrix = np.zeros((len(users), len(movies)))

    #populate utility matrix with ratings
    for i in ratings.itertuples():
        utility_matrix[i[1] - 1, i[2] - 1] = i[3]

    m, n = utility_matrix.shape

    #set random P and Q with 10 factors
    P = np.random.rand(m, 10)
    Q = np.random.rand(10, n)

    #set gamma and learning rate
    lamda = 0.01
    gamma = 0.001




    #gradient descent
    for epoch in range(100):
        print(epoch)
        for row in ratings.itertuples():
            eui = row[3] - np.dot(P[row[1]-1, :], Q[:, row[2]-1])
            P[row[1]-1, :] = P[row[1]-1, :] + gamma * 2 * (eui * Q[:, row[2]-1] - lamda * P[row[1]-1, :])
            Q[:, row[2]-1] = Q[:, row[2]-1] + gamma * 2 * (eui * P[row[1]-1, :] - lamda * Q[:, row[2]-1])


    #predicted ratings
    pred = P @ Q

    #store predicted ratings in a csv file
    pd.DataFrame(pred).to_csv("./data/latent_factors.csv", header=None, index=None)


    return pred

    ### for later - improvement

    # # pt = s * v
    #
    # # we find the full energy and the minimal allowed energy required (80%)
    # energy = np.sum(np.square(s))
    # min_energy = 0.8 * energy
    #
    # s = np.diag(s)
    #
    # # record the number of the least significant singular values we can remove
    # singular_values = s.diagonal()
    # sv_number = len(singular_values)
    #
    # # remove tells us the number of singular values that can be made 0
    # remove = 0
    # # removed_energy tells the amount of energy lost after we remove a singular value
    # removed_energy = 0
    #
    # for i in range (sv_number - 1,0, -1):
    #     removed_energy = np.square(singular_values[i]) + removed_energy
    #     if energy - removed_energy < min_energy:
    #         break
    #     remove = remove + 1
    #
    # # we set the least significant singular values to 0
    # s[:,sv_number - remove: sv_number + 1] = 0
    #
    # # we find factorization matrix P transposed
    # q = s.dot(v)



    # users = [i for i in range(len(utility_matrix[0,:]))]
    # items = [i for i in range(len(utility_matrix[:, 0]))]
    #
    # #
    # print(rmse(utility_matrix, p@q))
    #
    # # we perform gradient descent and update the
    # for i in range(0,3):
    #     for n, m in zip(np.arange(len(utility_matrix[0,:])), np.arange(len(utility_matrix[:,0]))):
    #         err = utility_matrix[n, m] - np.dot(p[n,:], q[:, m])
    #         p[n, :] += 0.001 * (err * q[:, m] - 0.1 * p[n, :])
    #         q[:, m] += 0.001 * (err * p[n, :] - 0.1 * q[:, m])
    #
    #     print(rmse(utility_matrix, p@q))


#####
##
## FINAL PREDICTORS
##
#####

def predict_final(movies, users, ratings, predictions):
    # # users x movies matrix
    utility_matrix = np.zeros((len(users), len(movies)))

    #populate utility matrix with ratings
    for i in ratings.itertuples():
        utility_matrix[i[1] - 1, i[2] - 1] = i[3]

    mean_users = np.average(utility_matrix, axis=1, weights=(utility_matrix > 0))

    no_rating_indices = np.where(~utility_matrix.any(axis=0))[0]

    mean_movies = np.zeros(utility_matrix.shape[1])

    for i in range (0, utility_matrix.shape[1]) :
        if not(i in no_rating_indices):
            mean_movies[i] = np.average(utility_matrix[:,i], weights=(utility_matrix[:, i] > 0))

    mean_rating = np.average(utility_matrix, weights=(utility_matrix > 0))

    user_biases = mean_users - mean_rating

    movie_biases = mean_movies - mean_rating


    pass


def predict_randoms(movies, users, ratings, predictions):
    number_predictions = len(predictions)

    return [[idx, randint(1, 5)] for idx in range(1, number_predictions + 1)]


##########################################
# MAIN FUNCTION


### TESTING


#####
##
## SAVE RESULTS
##
#####


###################################
## commented out for later


## //!!\\ TO CHANGE by your prediction function


predictions = predict_latent_factors(movies_description, users_description, ratings_description, predictions_description)
# predictions = predict_collaborative_filtering(movies_description, users_description, predo,
#                                                predictions_description)

# predict collaborative filtering


# Save predictions, should be in the form 'list of tuples' or 'list of lists'
with open(submission_file, 'w') as submission_writer:
    # Formates data
    predictions = [map(str, row) for row in predictions]
    predictions = [','.join(row) for row in predictions]
    predictions = 'Id,Rating\n' + '\n'.join(predictions)

    # Writes it dowmn
    submission_writer.write(predictions)

print("end")
