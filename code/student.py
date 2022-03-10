import numpy as np
import cv2
import random

def calculate_projection_matrix(image, markers):
    """
    To solve for the projection matrix. You need to set up a system of
    equations using the corresponding 2D and 3D points. See the handout, Q5
    of the written questions, or the lecture slides for how to set up these
    equations.

    Don't forget to set M_34 = 1 in this system to fix the scale.

    :param image: a single image in our camera system
    :param markers: dictionary of markerID to 4x3 array containing 3D points
    :return: M, the camera projection matrix which maps 3D world coordinates
    of provided aruco markers to image coordinates
    """
    ######################
    # Do not change this #
    ######################

    # Markers is a dictionary mapping a marker ID to a 4x3 array
    # containing the 3d points for each of the 4 corners of the
    # marker in our scanning setup
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_1000)
    parameters = cv2.aruco.DetectorParameters_create()

    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(
        image, dictionary, parameters=parameters)
    markerIds = [m[0] for m in markerIds]
    markerCorners = [m[0] for m in markerCorners]

    points2d = []
    points3d = []

    for markerId, marker in zip(markerIds, markerCorners):
        if markerId in markers:
            for j, corner in enumerate(marker):
                points2d.append(corner)
                points3d.append(markers[markerId][j])

    points2d = np.array(points2d)
    points3d = np.array(points3d)

    ########################
    # TODO: Your code here #
    ########################
    # This M matrix came from a call to rand(3,4). It leads to a high residual.
    len=np.shape(points2d)[0]
    ct = 0
    A = np.zeros((2*len,11))
    b = np.zeros((2*len,1))
    for i in range(len):
        X = points3d[i,0]
        Y = points3d[i,1]
        Z = points3d[i,2]
        u = points2d[i,0]
        v = points2d[i,1]
        row1 = [X,Y,Z,1,0,0,0,0,-u*X,-u*Y,-u*Z]
        row2 = [0,0,0,0,X,Y,Z,1,-v*X,-v*Y,-v*Z]
        A[ct,:] = row1
        b[ct,:] = u
        ct = ct + 1
        A[ct,:] = row2
        b[ct,:] = v
        ct = ct + 1  

    vec = np.linalg.lstsq(A, b, rcond=None)[0]
    A1 = np.transpose(vec[0:4])
    B1 = np.transpose(vec[4:8])
    C1 = vec[8:11]
    C1 = np.append(C1, 1)

    M = np.zeros((3, 4))
    M[0,:] = A1
    M[1,:] = B1
    M[2,:] = C1

    return M


def normalize_coordinates(points):
    """
    ============================ EXTRA CREDIT ============================
    Normalize the given Points before computing the fundamental matrix. You
    should perform the normalization to make the mean of the points 0
    and the average magnitude 1.0.

    The transformation matrix T is the product of the scale and offset matrices

    Offset Matrix
    Find c_u and c_v and create a matrix of the form in the handout for T_offset

    Scale Matrix
    Subtract the means of the u and v coordinates, then take the reciprocal of
    their standard deviation i.e. 1 / np.std([...]). Then construct the scale
    matrix in the form provided in the handout for T_scale

    :param points: set of [n x 2] 2D points
    :return: a tuple of (normalized_points, T) where T is the [3 x 3] transformation
    matrix
    """
    ########################
    # TODO: Your code here #
    ########################
    # This is a placeholder with the identity matrix for T replace with the
    # real transformation matrix for this set of points
    T = np.eye(3)

    return points, T


def estimate_fundamental_matrix(points1, points2):
    """
    Estimates the fundamental matrix given set of point correspondences in
    points1 and points2.

    points1 is an [n x 2] matrix of 2D coordinate of points on Image A
    points2 is an [n x 2] matrix of 2D coordinate of points on Image B

    Try to implement this function as efficiently as possible. It will be
    called repeatedly for part IV of the project

    If you normalize your coordinates for extra credit, don't forget to adjust
    your fundamental matrix so that it can operate on the original pixel
    coordinates!

    :return F_matrix, the [3 x 3] fundamental matrix
    """
    ########################
    # TODO: Your code here #
    ########################
    len = min(np.shape(points1)[0],np.shape(points2)[0])

    A = np.zeros((len,9))

    for i in range(len):
        u1 = points1[i,0]
        v1 = points1[i,1]
        u2 = points2[i,0]
        v2 = points2[i,1]
        row = [u1*u2,v1*u2,u2,u1*v2,v1*v2,v2,u1,v1,1]
        A[i,:] = row
    U, S, Vh = np.linalg.svd(A)
    F_matrix = Vh[-1]
    F_matrix = np.reshape(F_matrix, (3, 3))
    U, S, Vh = np.linalg.svd(F_matrix)
    S[-1] = 0
    S = np.diagflat(S)
    F_matrix = np.dot(np.dot(U,S),Vh) 
    # This is an intentionally incorrect Fundamental matrix placeholder
    # F_matrix = np.array([[0, 0, -.0004], [0, 0, .0032], [0, -0.0044, .1034]])

    return F_matrix


def ransac_fundamental_matrix(matches1, matches2, num_iters):
    """
    Find the best fundamental matrix using RANSAC on potentially matching
    points. Run RANSAC for num_iters.

    matches1 and matches2 are the [N x 2] coordinates of the possibly
    matching points from two pictures. Each row is a correspondence
     (e.g. row 42 of matches1 is a point that corresponds to row 42 of matches2)

    best_Fmatrix is the [3 x 3] fundamental matrix, inliers1 and inliers2 are
    the [M x 2] corresponding points (some subset of matches1 and matches2) that
    are inliners with respect to best_Fmatrix

    For this section, use RANSAC to find the best fundamental matrix by randomly
    sampling interest points. You would call the function that estimates the 
    fundamental matrix (either the "cheat" function or your own 
    estimate_fundamental_matrix) iteratively within this function.

    If you are trying to produce an uncluttered visualization of epipolar lines,
    you may want to return no more than 30 points for either image.

    :return: best_Fmatrix, inliers1, inliers2
    """
    # DO NOT TOUCH THE FOLLOWING LINES
    random.seed(0)
    np.random.seed(0)
    
    ########################
    # TODO: Your code here #
    ########################

    # Your RANSAC loop should contain a call to 'estimate_fundamental_matrix()'
    # that you wrote for part II.

    best_Fmatrix = estimate_fundamental_matrix(matches1[0:9, :], matches2[0:9, :])
    inliers_a = matches1[0:29, :]
    inliers_b = matches2[0:29, :]

    return best_Fmatrix, inliers_a, inliers_b


def matches_to_3d(points1, points2, M1, M2):
    """
    Given two sets of points and two projection matrices, you will need to solve
    for the ground-truth 3D points using np.linalg.lstsq(). For a brief reminder
    of how to do this, please refer to Question 5 from the written questions for
    this project.


    :param points1: [N x 2] points from image1
    :param points2: [N x 2] points from image2
    :param M1: [3 x 4] projection matrix of image2
    :param M2: [3 x 4] projection matrix of image2
    :return: [N x 3] list of solved ground truth 3D points for each pair of 2D
    points from points1 and points2
    """
    points3d = []
    ########################
    # TODO: Your code here #
    ########################
    len = np.shape(points1)[0]
    points3d = np.zeros((len,3))
    m11=M1[0,0]
    m12=M1[0,1]
    m13=M1[0,2]
    m14=M1[0,3]
    m21=M1[1,0]
    m22=M1[1,1]
    m23=M1[1,2]
    m24=M1[1,3]
    m31=M1[2,0]
    m32=M1[2,1]
    m33=M1[2,2]
    m34=M1[2,3]
    n11=M2[0,0]
    n12=M2[0,1]
    n13=M2[0,2]
    n14=M2[0,3]
    n21=M2[1,0]
    n22=M2[1,1]
    n23=M2[1,2]
    n24=M2[1,3]
    n31=M2[2,0]
    n32=M2[2,1]
    n33=M2[2,2]
    n34=M2[2,3]

    a=np.array([[m31-m11,m32-m12,m33-m13],[m31-m21,m32-m22,m33-m23],[n31-n11,n32-n12,n33-n13],[n31-n21,n32-n22,n33-n23]])
    for i in range(len):
        u1=points1[i,0]
        v1=points1[i,1]
        u2=points2[i,0]
        v2=points2[i,1]
        b=[m14-m34*u1,m24-m34*v1,n14-n34*u2,n24-n34*v2]
        points3d[i,:] = np.linalg.lstsq(a, b,rcond=None)[0]      
    points3d = points3d.tolist()
    return points3d
