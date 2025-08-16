from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np


json_fle = open('model.json','r')
loaded_file = json_fle.read()
json_fle.close()
model = model_from_json(loaded_file)
model.load_weights('model.weights.h5')
print("Model loaded successfully......")



def classify(img_file):
    img = img_file
    test_img = image.load_img(img,target_size=(128,128))
    test_img = image.img_to_array(test_img)
    test_img = test_img/255.0
    test_img = np.expand_dims(test_img,axis=0)
    
    result = model.predict(test_img)
    
    arr = np.array(result[0])
    print(arr)
    
    labels = ["Yna",'Tamatar','Thaa','Daa',"Dhaa","Adna","Tabla","Tha","Da","Dha",'ka',"Na",'Pa',
                 'Pha',"Ba","Bha","Ma","Yaw","Ra","La","Waw","Kha","Motosaw","petchr=iryakha","patalosa","Ha","Chyya",
                 'Tra','gya',"Ga","Gha",'Kna','Cha',"Chha","Ja","Jha"]
    
    
    
    prob = int(arr.argmax())
    
    prediction = labels[prob]
    
    print(prediction,":",img)
    
import os
files = []

path = "Dataset/DevanagariHandwrittenCharacterDataset/chk"
for r,d,f in os.walk(path):
    for file in f :
        files.append(os.path.join(r,file))
        
for f in files :
   classify(f)
   print()
 
#path = "Dataset/DevanagariHandwrittenCharacterDataset/Train/character_11_taamatar/192.png"
#classify(path)