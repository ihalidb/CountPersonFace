import pygame
import pathlib
import cv2
import numpy as np

video = cv2.VideoCapture(0)

cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_alt2.xml"
faceDetect = cv2.CascadeClassifier(str(cascade_path))

pygame.init()

window = pygame.display.set_mode((1200, 700))

pygame.display.set_caption("Face Detection App")

start = True

while start:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            start = False
            pygame.quit()
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(frame, 1.3, 5)
    for (x, y, w, h) in faces:
        x1, y1 = x + w, y + h
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.line(frame, (x, y), (x + 30, y), (0, 0, 255), 6)  # Top Left
        cv2.line(frame, (x, y), (x, y + 30), (0, 0, 255), 6)

        cv2.line(frame, (x1, y), (x1 - 30, y), (0, 0, 255), 6)  # Top Right
        cv2.line(frame, (x1, y), (x1, y + 30), (0, 0, 255), 6)

        cv2.line(frame, (x, y1), (x + 30, y1), (0, 0, 255), 6)  # Bottom Left
        cv2.line(frame, (x, y1), (x, y1 - 30), (0, 0, 255), 6)

        cv2.line(frame, (x1, y1), (x1 - 30, y1), (0, 0, 255), 6)  # Bottom right
        cv2.line(frame, (x1, y1), (x1, y1 - 30), (0, 0, 255), 6)
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgRGB = np.rot90(imgRGB)
    imgRGB = pygame.surfarray.make_surface(imgRGB).convert()

    font = pygame.font.Font("BebasNeue-Regular.ttf", 50)
    text = font.render("{} Yüz Tespit Edildi".format(len(faces)), True, (255, 0, 0))

    window.blit(imgRGB, (280, 95))
    window.blit(text, (320, 320))
    pygame.display.update()
