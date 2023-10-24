import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from collections import Counter

def load_camera_param(file_name):
    # Carregar os parâmetros de calibração e mapa de retificação
    cv_file = cv.FileStorage(f"{file_name}", cv.FILE_STORAGE_READ)
    stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
    stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
    stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
    stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()
    Q = cv_file.getNode('q').mat()
    cv_file.release()
    return stereoMapL_x, stereoMapL_y, stereoMapR_x, stereoMapR_y

def remap_images(image_left, image_right, stereoMapL_x, stereoMapL_y, stereoMapR_x, stereoMapR_y):
    imgL = cv.imread(f'{image_left}', cv.COLOR_BGR2GRAY)
    imgR = cv.imread(f'{image_right}', cv.COLOR_BGR2GRAY)

    # Remapear as imagens usando o mapa de retificação
    rectifiedL = cv.remap(imgL, stereoMapL_x, stereoMapL_y, cv.INTER_LANCZOS4)
    rectifiedR = cv.remap(imgR, stereoMapR_x, stereoMapR_y, cv.INTER_LANCZOS4)
    return rectifiedL, rectifiedR

def sift_correspondence(image_left, image_right):
    # Configurar o algoritmo SIFT
    sift = cv.SIFT_create()

    # Encontrar keypoints e calcular descritores SIFT
    kpL, descL = sift.detectAndCompute(image_left, None)
    kpR, descR = sift.detectAndCompute(image_right, None)

    # Realizar correspondência de keypoints
    bf = cv.BFMatcher()
    matches = bf.knnMatch(descL, descR, k=2)

    # Filtrar correspondências usando o teste de razão de Lowe
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    return good_matches, kpL, kpR

def orb_correspondence(image_left, image_right):
    # Configurar o algoritmo ORB
    orb = cv.ORB_create()

    # Encontrar keypoints e calcular descritores ORB
    kpL, descL = orb.detectAndCompute(image_left, None)
    kpR, descR = orb.detectAndCompute(image_right, None)

    # Realizar correspondência de keypoints
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)  # Usar Hamming distance para descritores binários
    matches = bf.match(descL, descR)

    # Filtrar correspondências
    good_matches = []
    threshold = 50
    for match in matches:
        if match.distance < threshold:
            good_matches.append(match)

    return good_matches, kpL, kpR

def brisk_correspondence(image_left, image_right):
    # Configurar o algoritmo BRISK
    brisk = cv.BRISK_create()

    # Encontrar keypoints e calcular descritores BRISK
    kpL, descL = brisk.detectAndCompute(image_left, None)
    kpR, descR = brisk.detectAndCompute(image_right, None)

    # Realizar correspondência de keypoints
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)  # Usar Hamming distance para descritores binários
    matches = bf.match(descL, descR)

    # Filtrar correspondências
    good_matches = []
    threshold = 50
    for match in matches:
        if match.distance < threshold:
            good_matches.append(match)

    return good_matches, kpL, kpR

def triangulation(good_matches, kpL, kpR, image_colored):
    # Inicializar arrays para pontos 3D, pontos 2D correspondentes e cores
    points_3d = []
    points_2dL = []
    colors = []
    list_z = []
    image_color= cv.imread(f'{image_colored}', cv.COLOR_BGR2GRAY)

    baseline = 6.2
    focal_lenght = 917.7

    for match in good_matches:
        ptL = kpL[match.queryIdx].pt
        ptR = kpR[match.trainIdx].pt
        disparity = abs(ptL[0] - ptR[0])
        print(f"disparity: {disparity}")
        depth = round((baseline*focal_lenght)/disparity)
        list_z.append(depth)
        color = image_color[int(ptL[1]), int(ptL[0])]
        points_3d.append([ptL[0], ptL[1], abs(depth)])
        points_2dL.append(ptL)
        colors.append(color)

    # Converter as listas em matrizes numpy
    points_3d = np.array(points_3d)
    colors = np.array(colors)
    print(f"Quantidade de Pontos: {len(good_matches)}")
    print(f"Profundidade que mais ocorre: {Counter(list_z)}")

    return points_3d, colors

def generate_point_cloud(points_3d, colors):
    # Gerando Nuvem de pontos
    with open("point_cloud.ply", "w") as plyfile:
        plyfile.write("ply\n")
        plyfile.write("format ascii 1.0\n")
        plyfile.write("element vertex {}\n".format(len(points_3d)))
        plyfile.write("property float x\n")
        plyfile.write("property float y\n")
        plyfile.write("property float z\n")
        plyfile.write("property uchar red\n")
        plyfile.write("property uchar green\n")
        plyfile.write("property uchar blue\n")
        plyfile.write("end_header\n")
        for i in range(len(points_3d)):
            point = points_3d[i]
            color = colors[i]
            plyfile.write("{} {} {} {} {} {}\n".format(point[0], -point[1], point[2], color[2], color[1], color[0]))

    print("point cloud saved as point_cloud.ply")

if __name__ == "__main__":
    stereoMapL_x, stereoMapL_y, stereoMapR_x, stereoMapR_y = load_camera_param('stereoMap.xml')
    image_left, image_right = remap_images('CasosCubo\Caso10\printL0.png', 'CasosCubo\Caso10\printR0.png', stereoMapL_x, stereoMapL_y, stereoMapR_x, stereoMapR_y)
    good_matches, kpL, kpR = sift_correspondence(image_left, image_right)
    # good_matches, kpL, kpR = orb_correspondence(image_left, image_right)
    # good_matches, kpL, kpR = brisk_correspondence(image_left, image_right)
    points_3d, colors = triangulation(good_matches, kpL, kpR, 'CasosCubo\Caso10\printL0.png')
    generate_point_cloud(points_3d, colors)