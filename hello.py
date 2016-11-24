import os
from flask import Flask, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
#import cv2
import numpy as np 
from keras.models import load_model
from skimage import io


"""
#--------------------------------------
from keras.layers import Input,Dense,Convolution2D,MaxPooling2D,UpSampling2D,AveragePooling2D
from keras.models import Model
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Convolution2D,MaxPooling2D
from keras.layers import Convolution3D,MaxPooling3D
from keras.layers.convolutional import UpSampling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import Sequential
import numpy as np
from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras import callbacks
from keras.layers.convolutional import Deconvolution2D
from keras.layers.core import Reshape
from keras.models import load_model


SHAPE_X=250
SHAPE_Y=250
NUM_INPUT=17


inp_shape=(3,SHAPE_Y,SHAPE_X)
model=Sequential()
model.add(Convolution2D(64,3,3,activation='sigmoid',input_shape=inp_shape,border_mode='same'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=None, border_mode='same', dim_ordering='default'))
model.add(Convolution2D(16,3,3,activation='sigmoid',input_shape=inp_shape,border_mode='same'))
model.add(UpSampling2D(size=(2, 2), dim_ordering='default'))
model.add(Convolution2D(6,3,3,activation='sigmoid',input_shape=inp_shape,border_mode='same'))
model.add(Convolution2D(16,3,3,activation='sigmoid',input_shape=inp_shape,border_mode='same'))
model.add(Convolution2D(3,3,3,activation='sigmoid',border_mode='same'))
model.summary()
model.compile(optimizer='adamax',loss='mean_squared_error')
"""
#----------------

try:
    model=load_model("model1.m")
except:
    model=load_model("model1.m")
    model.summary()










app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
filename=""



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS





@app.route("/X")
def hello():
        return "Hello World!"






@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print filename
            #I=cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            I = io.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print "ASDADAS",I[0:250,0:250].shape
            I=I[0:250,0:250]
            I=np.transpose(I)
        
            I=I.reshape(1,3,250,250)
            #I1=model.predict(I)
            I1=I
            print I1.shape
            I1=I1.transpose()
            I1=I1.reshape((250,250,3))
            I1=I1*255
            I1=I1.astype("uint8")
            print "ASDAD",I1.shape


            
          






           

            outfile = os.path.join(app.config['UPLOAD_FOLDER'], "out_"+filename)  
            #cv2.imwrite(outfile, I1)
            io.imsave(outfile,I1)
            #return redirect(url_for('upload_file',filename=filename))
            return ""+ \
            "<!DOCTYPE html>" + \
            "<html>" + \
            "<head>" + \
            "<title>Upload new File</title>" + \
            "</head>" + \
            "<body>" + \
            "<h1>Output File</h1>" + \
            "<img src=\"uploads/out_"+filename+"\" alt=\"output image\" />" + \
            "</html>"





    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''


@app.route('/uploads/<filename>')
def uploaded_file(filename):
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)










if __name__ == "__main__":
    app.run()
