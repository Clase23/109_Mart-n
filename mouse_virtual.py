import cv2
import math
import mediapipe as mp
from pynput.mouse import Button, Controller
import pyautogui

mouse = Controller()

cap = cv2.VideoCapture(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

(screen_width, screen_height) = pyautogui.size()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)

tipIds = [4, 8, 12, 16, 20]

pinch = False

def countFingers(image, hand_landmarks, handNo=0):

    if hand_landmarks:
        # Obtener todas las marcas de referencia en la primera mano visible
        landmarks = hand_landmarks[handNo].landmark

        # Contar dedos
        fingers = []

        for lm_index in tipIds:
                # Obtener los valores de la psosición "y" de la punta y parte inferior del dedo
                finger_tip_y = landmarks[lm_index].y 
                finger_bottom_y = landmarks[lm_index - 2].y

                # Verificar si algun dedo está abierto o cerrado
                if lm_index !=4:
                    if finger_tip_y < finger_bottom_y:
                        fingers.append(1)
                        # print("El dedo con ID ",lm_index," está abierto.")

                    if finger_tip_y > finger_bottom_y:
                        fingers.append(0)
                        # print("El dedo con ID ",lm_index," está cerrado.")

        totalFingers = fingers.count(1)

        # Pellizco

        finger_tip_x = int((landmarks[8].x))
        finger_tip_y = int((landmarks[8].y))

        thumb_tip_x = int((landmarks[4].x)*width)
        thumb_tip_y = int((landmarks[4].y)*height)

        cv2.line(image, (finger_tip_x, finger_tip_y),(thumb_tip_x, thumb_tip_y),(255,0,0),2)


# Definir una función para
def drawHandLanmarks(image, hand_landmarks):

    # Dibujar conexiones entre los puntos de referencia
    if hand_landmarks:

      for landmarks in hand_landmarks:
               
        mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS)


while True:
    success, image = cap.read()

    image = cv2.flip(image, 1)
    
    # Detectar los puntos de referencia de las manos
    results = hands.process(image)

    # Obtener la posición de los puntos de referencia del resultado procesado
    hand_landmarks = results.multi_hand_landmarks

    # Dibujar puntos de referencia
    drawHandLanmarks(image, hand_landmarks)

    # Obtener la posición de los dedos de la mano
    ##################
    countFingers(image, hand_landmarks)
    ##################

    cv2.imshow("Controlador de medios", image)

    # Cerrar la ventana al presionar la barra espaciadora
    key = cv2.waitKey(1)
    if key == 32:
        break

cv2.destroyAllWindows()
