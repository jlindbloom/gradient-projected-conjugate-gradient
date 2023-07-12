import numpy as np
import scipy.sparse as sps



# Create test signal
def make_1d_signal(x):
    """Makes a 1D test signal.
    """

    def _make_1d_signal(x):
        if x < 0.2:
            return 0
        elif (x >= 0.2) and (x < 0.4):
            return 1 
        elif (x >= 0.4) and (x < 0.6):
            return 0
        elif (x >= 0.6) and (x < 0.8):
            return 2
        elif (x >= 0.8) and (x <= 1.0):
            return 0
        else:
            return 0
    
    vecfunc = np.vectorize(_make_1d_signal)

    return vecfunc(x)



def make_shepp_logan_image(resolution):
    """Builds some 2D Shepp-Logan data.
    """

    phantomData = np.zeros((resolution, resolution))
    fourier_n = int(resolution/2)

    for j in range(1, 2*fourier_n + 1):
        for k in range(1, 2*fourier_n + 1):
            x = (j-fourier_n-1)/fourier_n
            y = (k-fourier_n-1)/fourier_n
            xi = (x-0.22)*np.cos(0.4*np.pi) + y*np.sin(0.4*np.pi)
            eta = y*np.cos(0.4*np.pi) - (x-0.22)*np.sin(0.4*np.pi)

            z = 0
            if ( (x/0.69)**2 + (y/0.92)**2 ) <= 1:
                z = 2

            if ( (x/0.6624)**2 + ((y+.0184)/.874)**2 ) <= 1:
                z = z - 0.98

            if ( (xi/0.31)**2 + (eta/0.11)**2 ) <= 1:
                z = z - 0.8;

            xi = (x + 0.22)*np.cos(0.6*np.pi)+y*np.sin(0.6*np.pi);
            eta = y*np.cos(0.6*np.pi)-(x+0.22)*np.sin(0.6*np.pi);

            if ( (xi/0.41)**2 + (eta/0.16)**2 ) <= 1:
                z = z - 0.8

            if ( (x/0.21)**2 + ((y - 0.35)/0.25)**2 ) <= 1:
                z = z + 0.4

            if ( (x/.046)**2 + ((y - 0.1)/.046)**2 ) <= 1:
                z = z + 0.4

            if ( (x/.046)**2 + ((y + 0.1)/.046)**2 ) <= 1:
                z = z + 0.4

            if ( ((x + 0.08)/.046)**2 + ((y+.605)/.023)**2 ) <= 1:
                z = z + 0.4

            if ( (x/.023)**2 + ((y+.605)/.023)**2 ) <= 1:
                z = z + 0.4

            if ( ((x-.06)/.023)**2 + ((y+.605)/.046)**2 ) <= 1:
                z = z + 0.4

            phantomData[j-1,k-1] = z

    phantomData = phantomData.T
    phantomData = np.flip(phantomData)
    phantomData = np.fliplr(phantomData)

    return phantomData



def build_1d_first_order_grad(N, boundary="periodic"):
    """Constructs a SciPy sparse matrix that extracts the (1D) discrete gradient of an input signal.
    Boundary parameter specifies how to handle the boundary conditions.
    """
    
    assert boundary in ["none", "periodic", "zero"], "Invalid boundary parameter."
    
    d_mat = sps.eye(N)
    d_mat.setdiag(-1,k=-1)
    d_mat = d_mat.tolil()
    
    if boundary == "periodic":
        d_mat[0,-1] = -1
    elif boundary == "zero":
        pass
    elif boundary == "none":
        d_mat = d_mat[1:,:]
    else:
        pass
    
    return d_mat



def build_2d_first_order_grad(M, N, boundary="periodic"):
    """Constructs a SciPy sparse matrix that extracts the discrete gradient of an input image.
    Assumes periodic BCs. Input image should have original dimension (M,N), must be flattened
    to compute matrix-vector product. First set is horizontal gradient, second is vertical.
    """

    # Construct our differencing matrices
    d_mat_horiz = build_1d_first_order_grad(N, boundary=boundary)
    d_mat_vert = build_1d_first_order_grad(M, boundary=boundary)
    
    # Build the combined matrix
    eye_vert = sps.eye(M)
    d_mat_one = sps.kron(eye_vert, d_mat_horiz)
    
    eye_horiz = sps.eye(N)
    d_mat_two = sps.kron(d_mat_vert, eye_horiz)

    full_diff_mat = sps.vstack([d_mat_one, d_mat_two])
    
    return full_diff_mat

