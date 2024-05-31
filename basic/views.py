from django.shortcuts import render


from django.core.files.storage import FileSystemStorage

from keras.models import load_model
#from keras.preprocessing import image
import keras.utils as image
import json

img_h,img_w=256,256
class_names=['Early_blight','Late_blight','healthy']
model=load_model('./models/m.h5')

def welcome(request):
    return render(request,'main1.html')
def predict(request):
    print(request)
    
    fileobj=request.FILES['filepath']
    fs=FileSystemStorage()
    filepathname=fs.save(fileobj.name,fileobj)
    filepathname=fs.url(filepathname)
    testimage='.'+filepathname
    img= image.load_img(testimage ,target_size=(img_h,img_w))
    x=image.img_to_array(img)
    
    x=x.reshape(1,img_h,img_w,3)
    p=model.predict(x)

    import numpy as np
    predictedlabel=class_names[np.argmax(p[0])]



    context={'filepathname':filepathname ,'predictedlabel':predictedlabel}
    return render(request,'main1.html',context)