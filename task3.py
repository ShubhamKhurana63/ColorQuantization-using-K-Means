import numpy as np
import cv2
from matplotlib import pyplot as plt

np.random.seed(sum([ord(c) for c in "khurana5"]))

FILE_SAVING_PATH='C://Users/Shubham/Documents/UB/CVIP/myproj2/'
#for writing the image
def writeImage(image,name):
    cv2.imwrite(FILE_SAVING_PATH+name+'.jpg',image)

#for saving the plots
def saveImage(fig,name):
    fig.savefig(FILE_SAVING_PATH+name+'.jpg')
    plt.close(fig)

#for calculating the euclidian distance for x,y coordinate system
def calculateDistance(points,center):
    list1=[]
    for point in points: 
             value=np.sqrt((center[0]-point[0])**2+(center[1]-point[1])**2)
             list1.append(value)
    return np.asarray(list1)
#for calcualting the euclidian distance forx,y,z coordinate system
def calculateDistanceForThree(points,center):
    list1=[]
    for point in points: 
             value=np.sqrt((center[0]-point[0])**2+(center[1]-point[1])**2+(center[2]-point[2])**2)
             list1.append(value)
    return np.asarray(list1)

#applying kmeans 
def applyKmeans(samples,K,centroids,notBaboon):
    centroidsNew=np.zeros((centroids.shape[0],centroids.shape[1]),dtype=np.float32)#creating the matrix for keeping the new centroids
    centroidDistanceArray=np.zeros((samples.shape[0],centroids.shape[0]),dtype=np.float32)#matrix for placing the distance
    classificationVector=np.zeros(centroids.shape[0])#classification vector
    ctr=1
    plt.figure(1)
    #would be running for 20 iterations
    while ctr<=20:
        print('iteration starting',ctr)
        for i in range(0,K):              
            if notBaboon:#flag if the code is being run for first 3 parts of task
                centroidDistanceArray[:,i]=calculateDistance(samples,centroids[i])#will be getting distance for each sample from all centroids
            else:
                centroidDistanceArray[:,i]=calculateDistanceForThree(samples,centroids[i])#will be getting distance for each sample from all centroids
        classificationVector=np.argmin(centroidDistanceArray,axis=1)#getting the index for the minimum distance from the min distance matrix
        for i in range(0,K):
            clusters=samples[classificationVector==i]#getting the clusters on the basis of the cluster values
            
            print('---------------',len(clusters))
            centroidsNew[i]=np.mean(clusters,axis=0)
            if notBaboon:
                plt.scatter(clusters[:,0:1],clusters[:,1:2],marker='^',edgecolors=colors[i],facecolors=colors[i]);
                for point in clusters:
                     plt.annotate(point,(point[0],point[1])) 
   
        flag=np.array_equal(np.int32(centroids),np.int32(centroidsNew))  #would be checking if the centroids computed and the old centroids are equal(if convergence has happened)   
        if flag:
            print('---------------converged----------------')
            break
        else:
            for i in range(0,K):
                centroids[i]=centroidsNew[i]
        if notBaboon:#for task 3.1,3.2,3.3
            print('-------------classification vector------------',classificationVector)
            plt.figure(2)
            plt.scatter(centroids[:,0:1],centroids[:,1:2],color=['red','green','blue'])
            for coords in centroids:
                plt.annotate(coords,(coords[0],coords[1]))
            saveImage(plt.figure(1),'task3_iter'+str(ctr)+'_a')#saving the plots
            saveImage(plt.figure(2),'task3_iter'+str(ctr)+'_b')#saving the plots       
        ctr=ctr+1
    return centroids,classificationVector

def performColorQuantization(valueOfK):
    baboonImage=cv2.imread("C://Users/Shubham/Documents/UB/CVIP/proj2_data/data/baboon.jpg",1)#reading the colored image
    reshapedImage=np.reshape(baboonImage,(-1,3))#reshaping the image vector giving vector with dimension(262144 * 3)
    indices=np.random.randint(reshapedImage.shape[0],size=valueOfK)#fetching the random arrays(for centroids)
    centroids=reshapedImage[indices]
    centroids=np.float32(centroids)
    finalCentroids,classificationVector=applyKmeans(np.float32(reshapedImage),valueOfK,centroids,False)#applying kmeans
    for i in range(0,classificationVector.shape[0]):
        reshapedImage[i]=finalCentroids[classificationVector[i]]#placing the centroids in the image vector(for K color photgraph)
    image1=np.reshape(reshapedImage,(baboonImage.shape))
    writeImage(image1,'task_3_baboon_'+str(valueOfK))



X=[[5.9, 3.2],
[4.6,2.9],
[6.2,2.8],
[4.7,3.2],
[5.5,4.2],
[5.0,3.0],
[4.9,3.1],
[6.7,3.1],
[5.1,3.8],
[6.0 ,3.0]]
X_Array=np.asarray(X)
u1=[6.2,3.2]
u2=[6.6,3.7]
u3=[6.5,3.0]

centeroids=np.asarray([u1,u2,u3],dtype=np.float32)
colors=['red','green','blue']
applyKmeans(X_Array,3,centeroids,True)
performColorQuantization(3)
performColorQuantization(5)
performColorQuantization(10)
performColorQuantization(20)


