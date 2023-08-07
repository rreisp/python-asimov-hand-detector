import cv2
import mediapipe as mp
import numpy as np
import time                     # Check framerate

from typing import Union        # Union type; Union[X, Y] Significa ou X ou Y.

'''
Criando um módulo de tudo que aprendemos, para que não seja necessário repetir toooodo esse código quando formos usar
Utilizaremos apenas requisições
'''

# Class ===================
class AsimovDetector():
    def __init__(self, 
                    mode: bool = False, 
                    number_hands: int = 2, 
                    model_complexity: int = 1,
                    min_detec_confidence: float = 0.5, 
                    min_tracking_confidence: float = 0.5
                ):
        
        # Parametros necessário para inicializar o hands -> solução do mediapipe
        # https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
        self.mode = mode
        self.max_num_hands = number_hands
        self.complexity = model_complexity
        self.detection_con = min_detec_confidence
        self.tracking_con = min_tracking_confidence

        # Inicializando o hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode,
                                        self.max_num_hands,
                                        self.complexity,
                                        self.detection_con,
                                        self.tracking_con)    
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, 
                    img: np.ndarray, 
                    draw_hands: bool = True):
        # Correção de cor
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Coletando resultados do processo das hands e analisando-os
        self.results = self.hands.process(img_RGB)
        if self.results.multi_hand_landmarks:
            for hand in self.results.multi_hand_landmarks:
                if draw_hands:
                    self.mp_draw.draw_landmarks(img, hand, self.mp_hands.HAND_CONNECTIONS)  

        return img

    def find_position(self, 
                        img: np.ndarray, 
                        hand_number: int = 0, 
                        draw_hands: bool = True):
        self.required_landmark_list = []
        
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_number]
            for id, lm in enumerate(my_hand.landmark):
                height, width, _ = img.shape
                center_x, center_y = int(lm.x*width), int(lm.y*height)

                self.required_landmark_list.append([id, center_x, center_y])  

        return self.required_landmark_list
        
# Main para teste de classe
if __name__ == '__main__':
    # coletando o framerate e capturando o vídeo
    previous_time = 0
    current_time = 0
    capture = cv2.VideoCapture(1)

    Detector = AsimovDetector()

    while True:
        _, img = capture.read()
        
        img = Detector.find_hands(img) #, draw_hands=False)
        # landmark_list = Detector.find_position(img) #, draw_hands=False)
        # if landmark_list:
        #     print(landmark_list[8])

        current_time = time.time()
        fps = 1/(current_time - previous_time)      # Numero de Frames/Tempo retorna o numero de frames por segundo
        previous_time = current_time

        # img, text, coordenadas de origem, Fonte, TamanhoDaFonte, Cor, Grossura
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_DUPLEX, 2, (255,0,255), 3)

        cv2.imshow("Image", img)

        # Não é necessário focar nessa explicação MAS:
        # cv2.waitKey() retorna um 32 Bit integer (pode depender da plataforma). 
        # O key input (input do teclado) é um ASCII de 8 Bit integer value. Então tu só precisa se preocupar com esses 8, os outros podem ser zero. 
        # 0xFF é uma máscara pros 8bits finais
        # é uma bitwise operation
        # É possível alcançar isso com: 
        if cv2.waitKey(20) & 0xFF==ord('q'):
            break
