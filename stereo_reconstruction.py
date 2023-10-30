import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from collections import Counter


def load_camera_param(file_name):
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

    rectifiedL = cv.remap(imgL, stereoMapL_x, stereoMapL_y, cv.INTER_LANCZOS4)
    rectifiedR = cv.remap(imgR, stereoMapR_x, stereoMapR_y, cv.INTER_LANCZOS4)
    return rectifiedL, rectifiedR


def sift_correspondence(image_left, image_right):
    sift = cv.SIFT_create()

    kpL, descL = sift.detectAndCompute(image_left, None)
    kpR, descR = sift.detectAndCompute(image_right, None)

    bf = cv.BFMatcher()
    matches = bf.knnMatch(descL, descR, k=2)

    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    return good_matches, kpL, kpR


def orb_or_brisk_correspondence(image_left, image_right, algorithm='ORB', threshold=50):
    if algorithm == 'ORB':
        feature_detector = cv.ORB_create()
    elif algorithm == 'BRISK':
        feature_detector = cv.BRISK_create()
    else:
        raise ValueError("Algoritmo não especificado")

    kpL, descL = feature_detector.detectAndCompute(image_left, None)
    kpR, descR = feature_detector.detectAndCompute(image_right, None)

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descL, descR)

    good_matches = [match for match in matches if match.distance < threshold]

    return good_matches, kpL, kpR


def triangulation(good_matches, kpL, kpR, image_colored):
    points_3d = []
    points_2dL = []
    colors = []
    list_z = []
    image_color = cv.imread(f'{image_colored}', cv.COLOR_BGR2GRAY)

    baseline = 6.2
    focal_length = 917.7

    for match in good_matches:
        ptL = kpL[match.queryIdx].pt
        ptR = kpR[match.trainIdx].pt
        disparity = abs(ptL[0] - ptR[0])
        depth = round((baseline * focal_length) / disparity)
        list_z.append(depth)
        color = image_color[int(ptL[1]), int(ptL[0])]
        points_3d.append([ptL[0], ptL[1], abs(depth)])
        points_2dL.append(ptL)
        colors.append(color)

    points_3d = np.array(points_3d)
    colors = np.array(colors)
    most_depths = Counter(list_z).most_common(1)
    print(f"Quantidade de Pontos: {len(good_matches)}")

    return points_3d, colors, most_depths


def matches_and_depth_exibition(image_left, image_right, good_matches, most_depths, kpL, kpR):
    img_matches = cv.drawMatches(image_left, kpL, image_right, kpR,good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.putText(img_matches, f"Depth: {most_depths[0][0]} cm", (20, 450), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv.imshow('Matches and Depth', img_matches)
    cv.waitKey(0)
    cv.destroyAllWindows()


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
            plyfile.write("{} {} {} {} {} {}\n".format(
                point[0], -point[1], point[2], color[2], color[1], color[0]))

    print("point cloud saved as point_cloud.ply")


if __name__ == "__main__":
    stereoMapL_x, stereoMapL_y, stereoMapR_x, stereoMapR_y = load_camera_param('stereoMap.xml')
    image_left, image_right = remap_images('CasosCubo\Caso11\printL0.png', 'CasosCubo\Caso11\printR0.png', stereoMapL_x, stereoMapL_y, stereoMapR_x, stereoMapR_y)
    good_matches, kpL, kpR = sift_correspondence(image_left, image_right)
    # good_matches, kpL, kpR = orb_or_brisk_correspondence(image_left, image_right, 'BRISK')
    points_3d, colors, most_depths = triangulation(good_matches, kpL, kpR, 'CasosCubo\Caso10\printL0.png')
    matches_and_depth_exibition(image_left, image_right, good_matches, most_depths, kpL, kpR)
    generate_point_cloud(points_3d, colors)
