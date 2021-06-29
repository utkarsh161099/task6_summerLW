#Importing Libraries
import pandas as pd
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import os
import smtplib,time  
from email.message import EmailMessage
import pywhatkit
import imghdr  
 
 # Load HAAR face classifier
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#loadFunction
def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    if faces is ():
        return None
    
    # Crop all faces found
    for (x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face

# Initialize Webcam
cap = cv2.VideoCapture(0)
count = 0

# Collect 100 samples of your face from webcam input
while True:

    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count += 1
        face = cv2.resize(face_extractor(frame), (200, 200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Save file in specified directory with unique name
        file_name_path = './faces/user/' + str(count) + '.jpg'
        cv2.imwrite(file_name_path, face)

        # Put count on images and display live count
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Face Cropper', face)
        
    else:
        print("Face not found")
        pass

    if cv2.waitKey(1) == 13 or count == 100: #13 is the Enter Key
        break
        
cap.release()
cv2.destroyAllWindows()      
print("Collecting Samples Complete")
Collecting Samples Complete
cap.release()
from os import listdir
from os.path import isfile, join

# Get the training data we previously made
data_path = './faces/user/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

# Create arrays for training data and labels
Training_Data, Labels = [], []

# Open training images in our datapath
# Create a numpy array for training data
for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)

# Create a numpy array for both training data and labels
Labels = np.asarray(Labels, dtype=np.int32)

# Initialize facial recognizer
# model = cv2.face.createLBPHFaceRecognizer()
# NOTE: For OpenCV 3.0 use cv2.face.createLBPHFaceRecognizer()
# pip install opencv-contrib-python
# model = cv2.createLBPHFaceRecognizer()

utkarsh_model  = cv2.face_LBPHFaceRecognizer.create()
# Let's train our model 
utkarsh_model.train(np.asarray(Training_Data), np.asarray(Labels))
print("Model trained sucessefully")
Model trained sucessefully
import os


face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_detector(img, size=0.5):
    
    # Convert image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return img, []
    
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))
    return img, roi


# Open Webcam
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    
    image, face = face_detector(frame)
    
    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Pass face to prediction model
        # "results" comprises of a tuple containing the label and the confidence value
        results = utkarsh_model.predict(face)
        # harry_model.predict(face)
        
        if results[1] < 500:
            confidence = int( 100 * (1 - (results[1])/400) )
            display_string = str(confidence) + '% Confident it is Utkarsh'
            
        cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255,120,150), 2)
        
        if confidence > 90:
            cv2.putText(image, "Hey Utkarsh", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Face Recognition', image )
            #os.system("chrome https://www.google.com/search?q=vimal+daga")
            #os.system("notepad")
            send_txt_mail()
            send_img_mail()
            
            #Running AWS instance
            #Make sure you have AWS CLI is configured in your system.
            
            #instance_id=sp.getoutput(" aws   ec2   run-instances --image-id   ami-0e306788ff2473ccb  --instance-type   t2.micro  --count  1  --subnet-id subnet-e3fdc78b   --security-group-ids  sg-07b3bc57a5baf2214   --key-name   mykey ")
            #ins=json.loads(instance_id)
            #volid=sp.getoutput("aws ec2 create-volume --availability-zone   ap-south-1a  --size  4   --volume-type   gp2")
            #VolId=json.loads(volid)
            #vol_attach=sp.getoutput("aws ec2 attach-volume --volume-id  {}    --instance-id  {}   --device /dev/xvdh".format(VolId['VolumeId'],ins['Instances'][0]['InstanceId']))
            
            break
         
        else:
            
            cv2.putText(image, "I dont know, who r u", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            cv2.imshow('Face Recognition', image )

    except:
        cv2.putText(image, "No Face Found", (220, 120) , cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.putText(image, "looking for face", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.imshow('Face Recognition', image )
        pass
        
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break
        
cap.release()
cv2.destroyAllWindows()     



# for mail configuration

def send_txt_mail():
    

    msg = EmailMessage()
    msg['subject'] = "Trial To Send Msg"
    msg['from'] = "xxxxx@gmail.com"
    msg['to'] = "xxxxx@gmail.com"
    msg.set_content(" Location Alert..! We've detected your face.")
    
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login("xxxxx@gmail.com", "xxxxxx") #add username and password!
        server.send_message(msg)
    print("Succes...!")
    
def send_img_mail():
    From = "Face_Detector_Machine Created By me...!!"
    Reciever_Email = "xxxxx@gmail.com"

    msg = EmailMessage()                         
    msg['Subject'] = "Your face has detected..!" 
    msg['From'] = From
    msg['To'] = Reciever_Email 

    time.sleep(3)


    with open('./faces/user/10.jpg', 'rb') as f:
        image_data = f.read()
        image_type = imghdr.what(f.name)
        image_name = f.name

    msg.add_attachment(image_data, maintype='image', subtype=image_type, filename=image_name)


    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login("xxxx@gmail.com", "xxxx")
        server.send_message(msg)
        
    print("Succes...!")

from twilio.rest import Client

def whatsappMsg():
    # client credentials are read from TWILIO_ACCOUNT_SID and AUTH_TOKEN
    client = Client()

    # this is the Twilio sandbox testing number
    from_whatsapp_number='whatsapp:xxxxxxxxxx'
    # replace this number with your own WhatsApp Messaging number
    to_whatsapp_number='xxxxxxxxx'

    client.messages.create(body='Face Detected',
                           from_=from_whatsapp_number,
                           to=to_whatsapp_number)
