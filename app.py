from flask import Flask, render_template, url_for, request, redirect, session, g
from datetime import datetime
import numpy as np
import array
import base64
from random import *
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from math import log10, sqrt
from PIL import Image
from matplotlib import cm
from os import getcwd
import statistics


PATH = getcwd() + '/static/images/'
IMAGE = []
app = Flask(__name__)
app.secret_key = "super secret key"




@app.route('/', methods=['POST', 'GET'])
def run():
    global IMAGE

    if "save" in request.form.keys(): 
        image_name = request.form['save']
        session["IMAGE_PATH"] = PATH + image_name
        if image_name.split(".")[1] == "pgm":
            IMAGE = read_pgm(session.get("IMAGE_PATH"))
            write_pgm(session.get("IMAGE_PATH"), IMAGE)
        elif image_name.split(".")[1] == "ppm":
            IMAGE = read_ppm(session.get("IMAGE_PATH"))
            write_ppm(session.get("IMAGE_PATH"), IMAGE)

    if "average" in request.form.keys() :
        apply_linear_filter(session.get("IMAGE_PATH"), moyenneur(3), "average")

    if len(IMAGE):
        #return display_image(IMAGE)
        png_format = PATH + "dummy.png"
        cv2.imwrite(png_format, IMAGE)
        return render_template("index.html", image=PATH+"dummy.png")
    return render_template("index.html", image="")


# ------------------------------------------ DISPLAY IN BASE64 ---------------------------------------------
# READ PPM IMAGE AND CONVERT IT TO BASE64 AND SEND IT TO IMG SRC IN HTML
def convert_to_base64(image):
    
    png_format = PATH + "dummy.png"
    cv2.imwrite(png_format, image)
    image = cv2.imread(png_format)
    image = image.tobytes()
    image = base64.b64encode(image)
    return image


def display_image(image):
    #encoded_image = convert_to_base64(image)
    png_format = PATH + "dummy.png"
    cv2.imwrite(png_format, image)
    image = cv2.imread(png_format)
    #image = "data:image/png;base64,"+str(encoded_image).strip()
    #image_url = url_for('image', image_name="dummy.png")
    return render_template("index.html", image=PATH+"dummy.png")

# ------------------------------------------ GET IMAGE ------------------------------------------------------
@app.route('/images/<image_name>')
def image(image_name):
    # Serve the image from the images directory
    return app.send_static_file('/static/images/' + image_name)

# --------------------------------------------- READ / WRITE -------------------------------------------------



def read_pgm(path):
    with open(path, 'r') as f :
        pgm = f.read()
    width, height = [int(x) for x in pgm.split('\n')[2].split()[:2]]
    image = np.array([x.split() for x in np.array(pgm.split('\n'))[4:]][0]).reshape(height, width)
    return image.astype('int32')


def read_ppm (path):
  with open(path, 'r') as f :
    ppm = f.read()
  image = np.array([x.split() for x in np.array(ppm.split('\n'))[5:385]])
  image=image.astype('int32')
  data=[]
  for row in image:
    row = row.reshape((row.shape[0]//3,3))
    data.append(row)
  data = np.array(data)
  return data


def write_pgm(path, image = []):
    if len(image):
        height = image.shape[0]
        width = image.shape[1]
        data = ""
        for i in range(0, height):
            for j in range(0, width):
                data += str(image[i][j]) + " "
        data +="\n"
        fout = open(path, "w")
        pgm_header = 'P2' + '\n' + str('#') + '\n' + str(width) + ' ' + str(height) + '\n' + str(255) + '\n'
        fout.write(pgm_header)
        fout.write(data)
        fout.close()


def write_ppm(path, image = []):
    if len(image): 
        data = ""
        for row in image:
            for element in row:
                data += str(int(element[0])) + " "+str(int(element[1])) + " "+str(int(element[2])) + " "
        data +="\n"
        fout = open(path, "w")
        pgm_header = 'P3' + '\n' + str(250) + ' ' + str(380) + '\n' + str(255) + '\n'
        fout.write(pgm_header)
        fout.write(data)
        fout.close()


# -------------------------------------------- FILTERS -------------------------------------------------

def apply_convolution(image, kernel):

    kernel = np.flipud(np.fliplr(kernel))    
    output = np.zeros_like(image)            
    
    image_padded = np.zeros((image.shape[0] + (kernel.shape[0]-1), 
                             image.shape[1] + (kernel.shape[1]-1)))   
    image_padded[(kernel.shape[0]//2):-(kernel.shape[0]//2), 
                 (kernel.shape[1]//2):-(kernel.shape[1]//2)] = image
    
    for x in range(image.shape[1]):     
        for y in range(image.shape[0]):
            
            output[y,x]=(kernel*image_padded[y:y + kernel.shape[0], x:x + kernel.shape[1]]).sum()
    return output




def moyenneur(size):
  return np.ones((size, size)) / size ** 2



def apply_linear_filter(path, filtre, filter_name=''):
  
  image = read_pgm(path)
  # print("original image : ", image)
  #image_bruit = bruit(image)
  # print("image bruitée : ", image_bruit) 
  #write_pgm(image.shape[0], image.shape[1], "noise_image.pgm", image_bruit)
  image_convolue = apply_convolution(image, filtre)
  # print("image convolue avec ce filtre : ", image_convolue)
  write_pgm('{}_image.pgm'.format(filter_name), image_convolue)

  return image_convolue




if __name__ == "__main__":
    app.run(debug=True)
