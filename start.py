import cv2, numpy as np, pandas as pd, datetime, os, face_recognition, glob
from tkinter import *
from tkinter.filedialog import *

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

        # Resize frame for a faster speed
        self.frame_resizing = 0.25

    def load_encoding_images(self, images_path):
        # Load Images ดึงชื่อไฟล์ทั้งหมดออกมาแล้วนำมาเชื่อมต่อกัน
        images_path = glob.glob(os.path.join(images_path, "*.*"))

        print("{} encoding images found.".format(len(images_path)))

        # Store image encoding and names
        for img_path in images_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Get the filename only from the initial file path.
            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)
            # Get encoding
            img_encoding = face_recognition.face_encodings(rgb_img)[0]

            # Store file name and file encoding
            self.known_face_encodings.append(img_encoding)
            self.known_face_names.append(filename)
        print("Encoding images loaded")

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        # Find all the faces and face encodings in the current frame of video
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name)

        # Convert to numpy array to adjust coordinates with frame resizing quickly
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names

root = Tk()
root.title('FaceRecognition')
root.geometry('800x600')

stdnames = []
realtime = []

def OpenCam():
    sfr = SimpleFacerec()
    sfr.load_encoding_images("images/")
    cap = cv2.VideoCapture(1)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    result = cv2.VideoWriter('CapCam.avi', fourcc, 20.0, (640,480))

    while True:
        ret, frame = cap.read()

        # Detect Faces
        face_locations, face_names = sfr.detect_known_faces(frame)
        for face_loc, name in zip(face_locations, face_names):
            y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 205, 50), 1)
            for x in range(len(name)):
                if name[x] not in stdnames:
                    pt = datetime.datetime.now()
                    dmy = pt.strftime('%d %b %Y %X %p')
                    realtime.append(dmy)
                    stdnames.append(name)

        Date = str(datetime.datetime.now())
        cv2.putText(frame, Date, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), cv2.LINE_4)
        cv2.imshow("FaceRecognition", frame)
        result.write(frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    result.release()
    cap.release()
    cv2.destroyAllWindows()

def showname():
    namelist = list(dict.fromkeys(stdnames))
    rst = list(dict.fromkeys(realtime))

    for nl in namelist:
        if nl == 'Unknown':
            namelist.remove('Unknown')

    datas = list(zip(namelist, rst))
    cols = ['Name', 'DateTime']
    df = pd.DataFrame(datas, columns=cols)
    df.to_csv('STDNAME.csv', index=False)

    fileopen = askopenfilename()
    file = open(fileopen, encoding='utf8')
    myLable = Label(text=file.read(), font=40).pack()

lb1 = Label(text='\n\n').pack()
btn1 = Button(root, text='Start Face Recognition', fg = 'yellow', font = 30, bg = 'black', command=OpenCam).pack()
lb2 = Label(text='\n').pack()
btn2 = Button(root, text='NameList', fg = 'yellow', font = 30, bg = 'black', command=showname).pack()

root.mainloop()