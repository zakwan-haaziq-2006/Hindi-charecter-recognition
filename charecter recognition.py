

from PyQt5 import QtCore, QtGui, QtWidgets
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.layers import Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np



class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setEnabled(True)
        self.label.setGeometry(QtCore.QRect(50, 60, 691, 61))
        self.label.setMinimumSize(QtCore.QSize(8, 5))
        self.label.setBaseSize(QtCore.QSize(6, 5))
        font = QtGui.QFont()
        font.setFamily("MS Sans Serif")
        font.setPointSize(13)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setLineWidth(6)
        self.label.setScaledContents(True)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(220, 130, 341, 241))
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(130, 420, 161, 51))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(130, 490, 161, 51))
        self.pushButton_2.setObjectName("pushButton_2")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(390, 450, 256, 61))
        self.textBrowser.setObjectName("textBrowser")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
        self.pushButton.clicked.connect(self.load_img)
        
        
        
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "HINDI CHARECTER RECOGNITION"))
        self.pushButton.setText(_translate("MainWindow", "BROWSE"))
        self.pushButton_2.setText(_translate("MainWindow", "CLASSIFY"))


    def load_img(self):
        filename,_ = QtWidgets.QFileDialog.getOpenFileName(None,"Select Image",
                            "","ImageFile(*.png *.jpg *.jpeg *.bmp);;All Files(*)")
        if filename :
            self.file = filename
            pixmap = QtGui.QPixmap(filename)
            pixmap = pixmap.scaled(self.label_2.width(),self.label_2.height())
            self.label_2.setPixmap(pixmap)
            self.label_2.setAlignment(QtCore.Qt.AlignCentre)
    
    
    def Classify(self):
        json_file = open("model.json")
        loaded_file = json_file.read()
        json_file.close()
        model = model_from_json(loaded_file)
        model.load_weights("model.weights.h5")
        print("Model loaded successfully....")
        
        label = ["Ka",'Kha','Ga','Gha',"Kna","Cha","Chha","Ja","Jha","Yna",'Taamatar',"Tha",'Daa',
                 'Dhaa',"Adna",'Tabala',"Tha","Da","Dha","Na","Pa","Pha","Ba","Bha","Ma","Yaw",
                 'Ra','La',"Waw","Motosaw",'Petchriyakha',"Ha","Chhya","Tra",'gya']
        path2 = self.file
        test_img = image.load_img(path2,target_size=(128,128))
        test_img = image.img_to_array(test_img)
        test_img = np.expand_dims(test_img)
        
        res = model.predict(test_img)
        arr = np.array(res)
        label2 = label[arr.argmax()]
        
        self.textEdit.setText(label2)
        
        
        
    def Training(self):
        model = Sequential()
        model.add(Conv2D(32,(3,3),activation='relu',input_shape = (128,128,1)))
        model.add(MaxPool2D((2,2)))
        model.add(BatchNormalization())
        
        model.add(Conv2D(64,(3,3),activation='relu'))
        model.add(MaxPool2D((2,2)))
        model.add(BatchNormalization())
        
        model.add(Conv2D(64,(3,3),activation='relu'))
        model.add(MaxPool2D((2,2)))
        model.add(BatchNormalization())
        
        
        model.add(Conv2D(96,(3,3),activation='relu'))
        model.add(MaxPool2D((2,2)))
        model.add(BatchNormalization())
        
        
        model.add(Conv2D(32,(3,3),activation='relu'))
        model.add(MaxPool2D((2,2)))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Flatten())
        
        model.add(Dense(128,activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(36,activation='softmax'))
        
        
        model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        
        train_datagen = ImageDataGenerator(rescale = None, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
        test_datagen = ImageDataGenerator(rescale = 1./255)
        
        training_set = train_datagen.flow_from_directory("Dataset/DevanagariHandwrittenCharacterDataset/Train",target_size = (128,128),batch_size = 8,class_mode = 'categorical')
        val_set = test_datagen.flow_from_directory("Dataset/DevanagariHandwrittenCharacterDataset/Test",target_size = (128,128),batch_size = 8,class_mode = 'categorical')
        
        
        model.fit(training_set,steps_per_epoch=100,epochs=10,validation_data=val_set,validation_steps=125)
        
        model_json = model.to_json()
        with open("model.josn",'w') as json_file :
            json_file.write(model_json)
        model.save_weights("model.weights.h5")
        print("model saved to disk")
    
   
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
