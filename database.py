import pymongo
import operator
import numpy as np
from keras.utils import Sequence

from models import *
from generator_28 import *

#Input : Image(256 x 256)
#Output : Corresponding_Vector(4096)

class DBMS(Sequence):
	def get_vector(self,Image,base_model):
	    Image = Image[np.newaxis,:,:,:]
	    Corresponding_Vector = base_model.predict(Image)
	    return Corresponding_Vector

	def form_dictionary(self,img_path,vector):
	    d={"Image_path":img_path}
	    d.update({'v'+ str(i):str(x) for i,x in enumerate(vector)})
	    return d


	def find_sum_diff(self,dict1,dict2):
	    
	    l=[]
	    for x in dict1:
		l.append(abs(float(dict1[x])-float(dict2[x])))
	    l = np.array(l)
	    mean_l = np.mean(l)
	    return mean_l


	def list_to_sorted_dictionary(self,l):
	    d=({i:x for i,x in enumerate(l)})
	    sorted_d = sorted(d.items(), key=operator.itemgetter(1))
	    print(len(sorted_d))
	    return sorted_d

	def Create_Database(self,Database_name,Collection_name):
	    myclient=pymongo.MongoClient()
	    mydb=myclient[Database_name]
	    mycol=mydb[Collection_name]
	    return mycol
        def return_from_list(self,L,idx):
	    D=[]
            for i in idx:
		D.append(L[i])
	    return D
	def Insert_Data(self,mycol,Image_path,Vector):
	    
	    threshold = 10
	    best_5 = []
	    image_path = []
	    sum_diff_list=[]
	    data = self.form_dictionary(Image_path,Vector[0])
	    y = data.copy()
	    y.pop('Image_path')
	    if mycol.count()==0: #Check if database is empty or not
	    	print("First Insertion.")
	    	mycol.insert_one(data)
	    	print("First Insertion Succesfull.")
	    else:
		sum_diff_list=[]
		for row in mycol.find():
	    	    x = row
	    	    img_p=x.pop('Image_path')
	    	    x.pop('_id')
	    	    sum_diff = self.find_sum_diff(x,y)
	    	    sum_diff_list.append(sum_diff)
                    image_path.append(img_p)
		    sorted_dictionary = self.list_to_sorted_dictionary(sum_diff_list)

		    if(sorted_dictionary[0][1] > threshold):
	   		mycol.insert_one(data)
		    else:
	    	        mycol.insert_one(data)
		        if(len(sorted_dictionary)>5):
                           for i in range(5):
		    	       best_5.append(sorted_dictionary[i][0])
	    		else:
			    for i in range(len(sorted_dictionary)):
		    	        best_5.append(sorted_dictionary[i][0])
		
		    print("The current image is : ",Image_path)
		    print("The best 5 matching paths are : \n")
		    paths = self.return_from_list(image_path,best_5)
		    diff = self.return_from_list(sum_diff_list,best_5)
		    for i in range(len(paths)):
			print("Path : ",paths[i]," Diff : ",diff[i])
		    print("================================================================")

                
            return(self.return_from_list(image_path,best_5),self.return_from_list(sum_diff_list,best_5))
