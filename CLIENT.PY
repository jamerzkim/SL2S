
import cv2
import socket as sk
import numpy as np
import threading

def vdisp():
    cam = cv2.VideoCapture('https://10.232.43.130:5000/video')
    s = sk.socket(sk.AF_INET,sk.SOCK_STREAM)
    s.bind((sk.gethostbyname(sk.gethostname()),5000))
    s.listen(10)
    conn, addr = s.accept()
    while True:
        data = conn.recv(90456)
        nparr = np.fromstring(data, np.uint8)
        decimg = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        _, frame = cam.read()
        cimg = cv2.resize(frame,(200,150),3)
        if type(decimg) is type(None):
            pass
        else:
            decimg[:150,:200] = cimg
            cv2.imshow('openvc',decimg)
            if cv2.waitKey(1) == 27:
                break
    cv2.destroyAllWindows()
    cam.release()

def vstream():
    cam = cv2.VideoCapture('https://10.232.43.130:5000/video')
    s = sk.socket(sk.AF_INET,sk.SOCK_STREAM)
    s.connect((sk.gethostbyname(sk.gethostname()),5000))
    while True:
        _, frame = cam.read()
        frame = cv2.resize(frame,(540,430))
        encimg = cv2.imencode(".jpg",frame)[1].tobytes()
        s.sendall(encimg)
    cam.release()
    cv2.destroyAllWindows()


disp = threading.Thread(target=vdisp)
stream = threading.Thread(target=vstream)
disp.start()
stream.start()