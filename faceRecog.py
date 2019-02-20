
# coding: utf-8

# In[3]:


import  sys
import dlib
import os
import cv2
import tensorflow as tf
import matplotlib


# In[4]:


#定义文件输入目录与输出目录
input_dir=r'C:/Users/biabiabia/Desktop/bs/data/face_recog/my_faces'
output_dir=r'C:/Users/biabiabia/Desktop/bs/data/my_faces'


# In[5]:


if not os.path.exists(output_dir):
    os.makedirs(output_dir)
detector=dlib.get_frontal_face_detector()


# In[ ]:


#预处理自己的头像
size=64
index=1
for(path,dirnames,filenames) in os.walk(input_dir):
    for filename in filenames:
        if filename.endswith('.jpg'):
            print('Being processed pircture %s' %index)
            img_path=path+r'/'+filename
            img=cv2.imread(img_path)   #读入一张图像 
            gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)   #将图片颜色空间转换到灰度空间
            dets=detector(gray_img,1)  #检测图像，返回一个列表，分别是矩形特征框的坐标函数。
            for i,d in enumerate(dets):
                x1=d.top() if d.top()>0 else 0
                y1=d.bottom() if d.bottom()>0 else 0
                x2=d.left() if d.left()>0 else 0
                y2=d.right() if d.right()>0 else 0
                face=img[x1:y1,x2:y2]
                face=cv2.resize(face,(size,size))
                cv2.imshow('image',face)   #显示face
                cv2.imwrite(output_dir+'/'+str(index)+'.jpg',face)   #把face存入输出文件夹中
                index+=1       
                key=cv2.waitKey(30)&0xff
                if key==27:        #27代表的是键盘esc键
                    sys.exit(0)

