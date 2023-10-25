import cv2 
import time

# Inicializa os detectores e descritores BRISK
brisk = cv2.BRISK_create()

# Inicializar o matcher de correspondência
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    # Capturar as imagens das lentes esquerda e direita
    suc, frame = cap.read()
    height, width = frame.shape[:2]
    frame_left = frame[:, :width//2]
    frame_right = frame[:, width//2:]
    
    img1 = frame_left
    img2 = frame_right

    start = time.time()

    # Converter as imagens para tons de cinza
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Detectar e descrever os pontos de interesse usando BRISK
    keypoints_1, descriptors_1 = brisk.detectAndCompute(img1_gray, None)
    keypoints_2, descriptors_2 = brisk.detectAndCompute(img2_gray, None)

    # Realizar a correspondência de recursos usando o matcher
    matches = bf.match(descriptors_1, descriptors_2)
    good_matches = [match for match in matches if match.distance < 50]

    # # Calcular profundidade e exibir na imagem
    # for match in good_matches:
    #     ptL = keypoints_1[match.queryIdx].pt
    #     ptR = keypoints_2[match.trainIdx].pt
    #     disparity = abs(ptL[0] - ptR[0])
    #     depth = round((baseline * focal_length) / disparity)

    end = time.time()
    totalTime = end - start

    fps = 1 / totalTime

    # Desenhar e exibir as correspondências encontradas
    img3 = cv2.drawMatches(img1_gray, keypoints_1, img2_gray, keypoints_2, matches, img2_gray, flags=2)
    cv2.putText(img3, f'BRISK FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
    # cv2.putText(img3, f'Depth: {depth}', (int(ptR[0]), int(ptR[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2))
    cv2.imshow('BRISK', img3)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        cv2.imwrite('BRISK.png', img3)
        break

cap.release()
cv2.destroyAllWindows()
