import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#euclidian distance calculator
def euclidian(x, y):
    dist = np.linalg.norm(x - y)
    return dist

#manhattan distance calculator
def manhattan(x, y):
    dist = np.sum(np.abs(x-y))
    return dist

def kFunction(X, y, k, functionChoice, norm):
    print(functionChoice,", k value: ", k, ", normalisation: ", norm)

    # iterations represent number of max iterations if convergence does not occur
    iterations = 100

    # if normalisation is required, normalise the samples
    if norm == "yes":
        X = X/ np.linalg.norm(X)

    # choose k number of random centroids from samples 
    centroids = X[np.random.choice(X.shape[0], size = k, replace=False), :]

    # iteration count for observing iteration of convergence
    iterationCount = 0

    for _ in range(iterations):
        iterationCount += 1

        # initialize empty clusters, empty cluster list for each k value    
        clusters = [[] for _ in range(k)]

        # assign cluster for each sample
        for idx, i in enumerate(X):
            # euclidian distance for kmeans
            if functionChoice == "kmeans":
                # get distance of each sample to each centroid
                distList = [euclidian(i, centroid) for centroid in centroids]
            
            # manhattan distance for kmedians
            elif functionChoice == "kmedians":
                distList = [manhattan(i, centroid) for centroid in centroids]

            # get centroid index of closest centroid to sample
            closestCentroidIndex = np.argmin(distList)

            # add sample index to cluster
            clusters[closestCentroidIndex].append(idx)
        
        #after updating cluster, store old centroids
        centroidsOld = centroids
        
        # initialise new centroids
        centroids = np.zeros((k, nFeatures))

        # for each cluster, calculate the mean/median and update centroid
        for clusterIdx, cluster in enumerate(clusters):
            if functionChoice == "kmeans":
                clusterAverage = np.mean(X[cluster], axis=0)
            elif functionChoice == "kmedians":
                clusterAverage = np.median(X[cluster], axis=0)
            # update centroids with calculated cluster mean/median
            centroids[clusterIdx] = clusterAverage

        #check if no update to centroid
        if functionChoice == "kmeans":
            error = [euclidian(centroidsOld[i], centroids[i]) for i in range(k)]
        elif functionChoice == "kmedians":
            error = [manhattan(centroidsOld[i], centroids[i]) for i in range(k)]

        # if centroids are the same, break the loop
        if np.sum(error) == 0:
            break

    #print(clusters)
    print("Iterations until convergence:", iterationCount)

    #############################################################################################################
    # get cluster labels

    # based on sample index in clusters, get sample labels
    sampleClass = [(np.array(y[cluster])).T for cluster in clusters]

    # create empty list of label class counter dictionaries for each cluster
    clusterCount = [{"0":0,"1":0, "2":0, "3":0} for i in sampleClass]

    # for each cluster, iterate through the sample labels and update dictionary count
    for idx, i in enumerate(sampleClass):
        for j in i[0]:
            if j == 0:
                clusterCount[idx]["0"] += 1
            elif j ==1:
                clusterCount[idx]["1"] += 1
            elif j ==2:
                clusterCount[idx]["2"] += 1
            else:
                clusterCount[idx]["3"] += 1
    
    #print(clusterCount)
    
    ############################################################################################################
    # get B-CUBED scores

    # initialise precision and recall
    precision = 0
    recall = 0

    # for each cluster count dictionary, calculate total precision and recall scores
    for idx, i in enumerate(clusterCount):
        precision += (i["0"] ** 2) /len(clusters[idx])
        precision += (i["1"] ** 2) /len(clusters[idx])
        precision += (i["2"] ** 2) /len(clusters[idx])
        precision += (i["3"] ** 2) /len(clusters[idx])
        recall += (i["0"] ** 2) / len(animals)
        recall += (i["1"] ** 2) / len(countries)
        recall += (i["2"] ** 2) / len(fruits)
        recall += (i["3"] ** 2) / len(veggies)
    
    # get average precision and recall
    precision = precision/len(X)
    recall = recall/len(X)
    # get fscore using calculated precision and recall
    fScore = 2*((precision*recall)/(precision+recall))

    print("Precision: ",precision)
    print("Recall: ", recall)
    print("Fscore:", fScore)
    print()
    return precision, recall, fScore

# function for plotting each question
def plotting(precision, recall, fScore, title):#, average):
    plt.title(title)
    plt.plot(kList, precision, marker='o', label = "precision", color="red")
    plt.plot(kList, recall, marker='o', label = "recall", color="blue")
    plt.plot(kList, fScore, marker='o', label = "fScore", color="green")
    #plt.plot(kList, average, marker='o', label = "average", color="purple")
    plt.xlabel("K value")
    plt.ylabel("Score")
    plt.legend()
    plt.show()

###################################################################################################
# format the data

# load data files
animals = pd.read_csv("animals", sep = " ", header = None).to_numpy()
countries = pd.read_csv("countries", sep = " ", header = None).to_numpy()
fruits = pd.read_csv("fruits", sep = " ", header = None).to_numpy()
veggies = pd.read_csv("veggies", sep = " ", header = None).to_numpy()

# label class data as 0,1,2,3
animals[:,0] = 0
countries[:,0] = 1
fruits[:,0] = 2
veggies[:,0] = 3

# combine all class data into one
data = np.vstack((animals, countries, fruits, veggies))

# seperate features and labels
X = data[:,1:]
y = data[:,:1]

# get number of features
nFeatures = len(X[0]) 

###############################################################################################################
#initialise variables based on question

np.random.seed(50)

#list of k values 1 to 9
kList = list(range(1,10))

# no for no normalisation, yes for normalisation
questions = [["kmeans", "no"],["kmeans", "yes"], ["kmedians", "no"], ["kmedians", "yes"]]

# for each question, cluster data, get precision/recall/fscore, plot results
for q in questions:
    #initialise empty lists for plotting
    precisionList = []
    recallList = []
    fscoreList = []
    functionChoice, norm = q
    title = str(functionChoice) + ", normalised: " + str(norm)
    
    # for each k value 1 to 9, append B-CUBED scores to list for plotting.
    for i in kList:
        precision, recall, fScore= kFunction(X, y, i, functionChoice, norm) 
        precisionList.append(precision)
        recallList.append(recall)    
        fscoreList.append(fScore)
        


    plotting(precisionList,recallList,fscoreList, title)

