#ライブラリ
import streamlit as st
import pandas as pd
import numpy as np
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import re
import os
import matplotlib.pyplot as plt
from sklearn import preprocessing
from PIL import Image

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh



#to modify aspect ratio
def aspect_normalize(file):
  #im = cv2.imread(file)
  as1 = file.shape[0]
  as2 = file.shape[1]
  return as1, as2 

#DataFrame
def dataframe_create(human_num_list, position_list, x_list, y_list, z_list):
  df = pd.DataFrame(human_num_list,columns=["human_num"])
  df["position_list"]=position_list
  df["x"]= x_list
  df["y"]= y_list
  df["z"]= z_list

  #normlize x,y
  x_list = []
  y_list = []
  z_list = []
  
  
  x_ = preprocessing.minmax_scale(df["x"].values)
  for k in x_:
    x_list.append(k) 
  y_ = preprocessing.minmax_scale(df["y"].values)
  for j in y_:
    y_list.append(j)
  z_ = preprocessing.minmax_scale(df["z"].values)
  for p in z_:
    z_list.append(p)
    
  df["x"] = x_list
  df["y"] = y_list
  df["z"] = z_list
  
  df_X_list = []
  df_Y_list = []
  df_Z_list = []

  df_X_ = df["x"].T
  df_Y_ = df["y"].T
  df_Z_ = df["z"].T

  df_X_list.append(df_X_.values)
  df_Y_list.append(df_Y_.values)
  df_Z_list.append(df_Z_.values)

  df_X = pd.DataFrame(df_X_list, columns = [ "X_{}".format(str(i).zfill(3)) for i in range(478)])
  df_Y = pd.DataFrame(df_Y_list, columns = [ "Y_{}".format(str(i).zfill(3)) for i in range(478)])
  df_Z = pd.DataFrame(df_Z_list, columns = [ "Z_{}".format(str(i).zfill(3)) for i in range(478)])

  # center alignment (fix nose_top on 0.5)
  gap_list = df_X["X_001"] -0.5
  X_list_rivise = []
  for i, k in enumerate(df_X.values):
    a = k - gap_list[i]
    X_list_rivise.append(a)
  df_X2 = pd.DataFrame(X_list_rivise, columns = [ "X_{}".format(str(i).zfill(3)) for i in range(478)])

  gap_list = df_Y["Y_001"] -0.5
  Y_list_rivise = []
  for i, k in enumerate(df_Y.values):
    a = k - gap_list[i]
    Y_list_rivise.append(a)
  df_Y2 = pd.DataFrame(Y_list_rivise, columns = [ "Y_{}".format(str(i).zfill(3)) for i in range(478)])

  df_XY = pd.concat([df_X2, df_Y2], axis = 1)

  return df_XY


#facemesh関数
def facemesh_photo(img_file):
  imgs_num_list=[]
  human_num_list =[]
  position_list=[]
  x_list=[]
  y_list=[]
  z_list=[]

  as1, as2 = aspect_normalize(img_file)  
  drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
  with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5) as face_mesh:
    #image = cv2.imread(img_file)
    results = face_mesh.process(cv2.cvtColor(img_file, cv2.COLOR_BGR2RGB))

    
      
    annotated_image = img_file.copy()
    blank = np.array(np.zeros(annotated_image.shape))

    human_num = 0
    for face_landmarks in results.multi_face_landmarks:
      
      for i in range(len(str(face_landmarks).split('landmark'))):
        cnt=0
        if str(face_landmarks).split('landmark')[i] != "":
          human_num_list.append(human_num)
          position_list.append(cnt)
          x_list.append(float(re.split('[:\n]',str(face_landmarks).split('landmark')[i])[2])/as1)
          y_list.append(float(re.split('[:\n]',str(face_landmarks).split('landmark')[i])[4])/as2)
          z_list.append(float(re.split('[:\n]',str(face_landmarks).split('landmark')[i])[6]))
          cnt+=1
      

      mp_drawing.draw_landmarks(
        image=blank,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_tesselation_style())
      
      human_num += 1

    df_XY = dataframe_create(human_num_list, position_list, x_list, y_list, z_list)
          
  return blank, df_XY


#本編  
st.title("Facemesh app")

img_source = st.radio("画像のソースを選択してください。",
                              ("画像をアップロード", "カメラで撮影"))
if img_source == "カメラで撮影":
  img_file_buffer = st.camera_input("カメラで撮影")
elif img_source == "画像をアップロード":
  img_file_buffer = st.file_uploader("ファイルを選択")
else:
    pass



  
if img_file_buffer :
  img_file_buffer_2 = Image.open(img_file_buffer)
  img_file = np.array(img_file_buffer_2)
  
  

  image_blank, df_XY = facemesh_photo(img_file)
  cv2.imwrite('temporary.jpg', image_blank)
  st.image('temporary.jpg')
  st.dataframe(df_XY)
  
  





