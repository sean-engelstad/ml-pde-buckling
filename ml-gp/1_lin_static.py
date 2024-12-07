# panel linear static inputs
# buckling inputs
# ----------------

def soln_error(eta, num_interior, num_bndry):

    import tensorflow as tf
    import numpy as np
    # import matplotlib.pyplot as plt

    print(f"Begin method : {num_interior=} {num_bndry=}", flush=True)

    # aluminum plate material properties
    E = 70e9; nu = 0.3; thick = 0.005
    D = E * thick**3 / 12.0 / (1 - nu**2)
    (a, b) = (3, 1)
    # (a, b) = (1,1)

    # choose kernel hyperparameters
    # Lx, Ly = (0.5, 0.4)
    Lx, Ly = (0.2, 0.2)
    # eta = 1e-9
    # eta = 1e-5

    def w_true(x,y):
        """true solution for verification"""
        return np.sin(np.pi * 3 * x) * np.sin(np.pi * 2 * y)

    # choose your own q_load
    # def q_load(x,y): # distributed load across the panel
    #     return np.sin(np.pi * 4 * x) * np.sin(np.pi * 3 * y)

    # choose q_load for method of manufactured solutions
    def q_load(x,y): # distributed load across the panel
        return 169.0 * D * w_true(x,y)

    # compute the linear static analysis inputs and mesh domain of collocation pts
    # ----------------------------------------------------------------------------


    # num_domain, num_bndry, num_test = (10, 5, 5)
    # num_domain, num_bndry, num_test = (500, 200, 50)
    num_test = 50
    # num_domain, num_bnry, num_test = (1000, 400, 50)
    # num_domain, num_bndry, num_test = (2000, 800, 50)
    # num_domain, num_bndry, num_test = (3600, 2000, 50)
    # num_domain, num_bndry, num_test = (3000, 600, 50)

    # num_interior = num_domain
    num_domain = num_interior
    DTYPE = tf.float32

    # set the boundary of the panel
    xmin = 0; xmax = a
    ymin = 0; ymax = b

    lb = tf.constant([xmin, ymin], dtype=DTYPE)
    ub = tf.constant([xmax, ymax], dtype=DTYPE)

    # generate the collocation points on the interior
    def gen_collocation_points(num_domain, minval, maxval):
        x_r = tf.random.uniform((num_domain,1), minval[0], maxval[0], dtype=DTYPE)
        y_r = tf.random.uniform((num_domain,1), minval[1], maxval[1], dtype=DTYPE)
        X_r = tf.concat([x_r, y_r], axis=1)
        # data_init = tf.random_uniform_initializer(minval=minval, maxval=maxval, seed=0)
        # return tf.Variable(data_init(shape=[num_domain, 1]), dtype=tf.float32)
        return tf.Variable(X_r, dtype=DTYPE)

    x_train = gen_collocation_points(num_domain, lb, ub)
    x_test = gen_collocation_points(num_test, lb, ub)
    # print(f"{type(x_train)=}")

    x = x_train[:,0:1]
    y = x_train[:,1:2]

    x2 = x_test[:,0:1]
    y2 = x_test[:,1:2]

    # generate boundary domain points
    assert num_bndry % 2 == 0
    N_b = int(num_bndry / 2)
    x_b1 = tf.random.uniform((N_b,1), lb[0], ub[0], dtype=DTYPE)
    y_b1 = lb[1] + (ub[1] - lb[1]) * \
        tf.keras.backend.random_bernoulli((N_b,1), 0.5, dtype=DTYPE)
    X_b1 = tf.concat([x_b1, y_b1], axis=1)

    # boundary data on x edges
    y_b2 = tf.random.uniform((N_b,1), lb[1], ub[1], dtype=DTYPE)
    x_b2 = lb[0] + (ub[0] - lb[0]) * \
        tf.keras.backend.random_bernoulli((N_b,1), 0.5, dtype=DTYPE)
    # print(f"{x_b2=}")
    X_b2 = tf.concat([x_b2, y_b2], axis=1)
    # print(f"{X_b2=}")
    # print(f"{x_b2=}")

    x_bndry = tf.Variable(tf.concat([X_b1, X_b2], axis=0), dtype=DTYPE)

    # plot the data to check
    # plt.scatter(x_bndry[:,0], x_bndry[:,1])
    # plt.scatter(x_train[:,0], x_train[:,1]) # only show 1000 of the points
    # plt.gca().set_aspect('equal')
    # plt.show()

    # q vector
    _temp = np.array([q_load(x[i], y[i]) for i in range(num_domain)])
    q = tf.constant(_temp, shape=(num_domain,1), dtype=DTYPE)

    # true solution
    # q vector
    _temp2 = np.array([w_true(x[i], y[i]) for i in range(num_domain)])
    w_true_vec = tf.constant(_temp2, shape=(num_domain,1), dtype=DTYPE)

    # define vectorized versions of the kernel functions
    # like 1000x faster at assembling covariance functions

    L_tf = tf.constant(np.array([Lx, Ly]), dtype=DTYPE)
    def SE_kernel2d_tf(x, xp, Lvec):
        # x input is N x 1 x 2 array, xp is 1 x M x 2 array
        # xbar is then an N x M x 2 shape array
        # print(f"{x=} {L_tf=}")
        xbar = (x - xp) / Lvec
        # output is N x M matrix of kernel matrix
        return tf.exp(-0.5 * tf.reduce_sum(tf.pow(xbar, 2.0), axis=-1))

    def kernel2d_tf(x, xp):
        """composite kernel definition"""
        return SE_kernel2d_tf(x, xp, L_tf)

    def d2_fact(xbar, L):
        return L**(-2.0) * (-1.0 + xbar**2)

    def d4_fact(xbar,L):
        return L**(-4.0) * (3.0 - 6.0 * xbar**2 + xbar**4)

    def d6_fact(xbar,L):
        return L**(-6.0) * (-15 + 45 * xbar**2 - 15 * xbar**4 + xbar**6)

    def d8_fact(xbar,L):
        return L**(-8.0) * (105 - 420 * xbar**2 + 210 * xbar**4 - 28 * xbar**6 + xbar**8)

    def SE_kernel2d_bilapl_tf(x, xp, Lvec):
        # first two dimensions are NxN matrix parts
        # x is N x 1 x 2 matrix, xp is 1 x M x 2 matrix
        xbar = (x - xp) / Lvec # N x M x 2 matrix
        x1bar = xbar[:,:,0]
        x2bar = xbar[:,:,1]
        Lx = Lvec[0]; Ly = Lvec[1]

        # baseline kernel matrix which we will scale and add some terms
        K = kernel2d_tf(x,xp)

        return K * (d4_fact(x1bar,Lx) + 2.0 * d2_fact(x1bar, Lx) * d2_fact(x2bar, Ly) + d4_fact(x2bar, Ly))

    def kernel2d_bilapl_tf(x, xp):
        """composite kernel definition"""
        return SE_kernel2d_bilapl_tf(x, xp, L_tf)

    def SE_kernel2d_double_bilapl_tf(x, xp, Lvec):
        # first two dimensions are NxN matrix parts
        # x is N x 1 x 2 matrix, xp is 1 x M x 2 matrix
        xbar = (x - xp) / Lvec # N x M x 2 matrix
        x1bar = xbar[:,:,0]
        x2bar = xbar[:,:,1]
        Lx = Lvec[0]; Ly = Lvec[1]

        # baseline kernel matrix which we will scale and add some terms
        K = kernel2d_tf(x,xp)

        return K * (d8_fact(x1bar,Lx) + \
                    4.0 * d6_fact(x1bar, Lx) * d2_fact(x2bar, Ly) +\
                    6.0 * d4_fact(x1bar, Lx) * d4_fact(x2bar, Ly) +\
                    4.0 * d2_fact(x1bar, Lx) * d6_fact(x2bar, Ly) +\
                    d8_fact(x2bar, Ly))

    def kernel2d_double_bilapl_tf(x, xp):
        """composite kernel definition"""
        return SE_kernel2d_double_bilapl_tf(x, xp, L_tf)

    # moment kernels
    def dx2_kernel_tf(x, xp):
        # first two dimensions are NxN matrix parts
        # x is N x 1 x 2 matrix, xp is 1 x M x 2 matrix
        xbar = (x - xp) / L_tf # N x M x 2 matrix
        x1bar = xbar[:,:,0]
        x2bar = xbar[:,:,1]
        Lx = L_tf[0]; Ly = L_tf[1]

        # baseline kernel matrix which we will scale and add some terms
        K = kernel2d_tf(x,xp)

        # TODO : shear or general case later (just axial now)
        return K * (d2_fact(x1bar, Lx))

    def dy2_kernel_tf(x, xp):
        # first two dimensions are NxN matrix parts
        # x is N x 1 x 2 matrix, xp is 1 x M x 2 matrix
        xbar = (x - xp) / L_tf # N x M x 2 matrix
        x1bar = xbar[:,:,0]
        x2bar = xbar[:,:,1]
        Lx = L_tf[0]; Ly = L_tf[1]

        # baseline kernel matrix which we will scale and add some terms
        K = kernel2d_tf(x,xp)
        return K * (d2_fact(x2bar, Ly))

    def doubledx2_kernel_tf(x, xp):
        # first two dimensions are NxN matrix parts
        # x is N x 1 x 2 matrix, xp is 1 x M x 2 matrix
        xbar = (x - xp) / L_tf # N x M x 2 matrix
        x1bar = xbar[:,:,0]
        x2bar = xbar[:,:,1]
        Lx = L_tf[0]; Ly = L_tf[1]

        # baseline kernel matrix which we will scale and add some terms
        K = kernel2d_tf(x,xp)

        # TODO : shear or general case later (just axial now)
        return K * (d4_fact(x1bar, Lx))

    def doubledy2_kernel_tf(x, xp):
        # first two dimensions are NxN matrix parts
        # x is N x 1 x 2 matrix, xp is 1 x M x 2 matrix
        xbar = (x - xp) / L_tf # N x M x 2 matrix
        x1bar = xbar[:,:,0]
        x2bar = xbar[:,:,1]
        Lx = L_tf[0]; Ly = L_tf[1]

        # baseline kernel matrix which we will scale and add some terms
        K = kernel2d_tf(x,xp)
        return K * (d4_fact(x2bar, Ly))

    def dx2_bilapl_kernel_tf(x, xp):
        # first two dimensions are NxN matrix parts
        # x is N x 1 x 2 matrix, xp is 1 x M x 2 matrix
        xbar = (x - xp) / L_tf # N x M x 2 matrix
        x1bar = xbar[:,:,0]
        x2bar = xbar[:,:,1]
        Lx = L_tf[0]; Ly = L_tf[1]

        # baseline kernel matrix which we will scale and add some terms
        K = kernel2d_tf(x,xp)

        # TODO : shear or general case later (just axial now)
        return K * \
            (d6_fact(x1bar,Lx) + 2.0 * d4_fact(x1bar, Lx) * d2_fact(x2bar, Ly) + d2_fact(x1bar, Lx) * d4_fact(x2bar, Ly))

    def dy2_bilapl_kernel_tf(x, xp):
        # first two dimensions are NxN matrix parts
        # x is N x 1 x 2 matrix, xp is 1 x M x 2 matrix
        xbar = (x - xp) / L_tf # N x M x 2 matrix
        x1bar = xbar[:,:,0]
        x2bar = xbar[:,:,1]
        Lx = L_tf[0]; Ly = L_tf[1]

        # baseline kernel matrix which we will scale and add some terms
        K = kernel2d_tf(x,xp)
        return K * \
            (d4_fact(x1bar,Lx) * d2_fact(x2bar, Ly) + 2.0 * d2_fact(x1bar, Lx) * d4_fact(x2bar, Ly) + d6_fact(x2bar, Ly))

    def dx2_dy2_kernel_tf(x, xp):
        # first two dimensions are NxN matrix parts
        # x is N x 1 x 2 matrix, xp is 1 x M x 2 matrix
        xbar = (x - xp) / L_tf # N x M x 2 matrix
        x1bar = xbar[:,:,0]
        x2bar = xbar[:,:,1]
        Lx = L_tf[0]; Ly = L_tf[1]

        # baseline kernel matrix which we will scale and add some terms
        K = kernel2d_tf(x,xp)
        return K * d2_fact(x1bar, Lx) * d2_fact(x2bar, Ly)

    # compute the block matrix Sigma = [cov(w_int, w_int), cov(w_int, w_bndry), cov(w_int, nabla^4 w_int),
    #                                   cov(w_bndry, w_int), cov(w_bndry, w_bndry), cov(w_bndry, nabla^4 w_bndry),
    #                                   cov(nabla^4 w_int, w_int), cov(nabla^4 w_int, w_bndry), cov(nabla^4 w_int, nabla^4 w_int) ]
    # not a function of theta..
    import tensorflow as tf
    import numpy as np
    import matplotlib.pyplot as plt

    n_block = 2 * num_interior + 3 * num_bndry
    x_all = tf.concat([x_train, x_bndry], axis=0)
    num_all = num_interior + num_bndry

    # components of the multivariate GP
    # 1 - w at interior and bndry points
    # 2 - d^2w/dx^2 at bndry points
    # 3 - d^2w/dy^2 at bndry points
    # 4 - nabla^4 w at interior points

    # 11 - interior+bndry self-covariance
    x_all_L = tf.expand_dims(x_all, axis=1)
    x_all_R = tf.expand_dims(x_all, axis=0)
    K11 = tf.constant(kernel2d_tf(x_all_L, x_all_R), dtype=DTYPE)

    x_interior_L = tf.expand_dims(x_train, axis=1)
    x_interior_R = tf.expand_dims(x_train, axis=0)
    x_bndry_L = tf.expand_dims(x_bndry, axis=1)
    x_bndry_R = tf.expand_dims(x_bndry, axis=0)

    # 12 - w with dx2
    K12 = tf.constant(dx2_kernel_tf(x_all_L, x_bndry_R))
    # 13 - w with dy2
    K13 = tf.constant(dy2_kernel_tf(x_all_L, x_bndry_R))
    # 14 - w with nabla^4 w
    K14 = tf.constant(kernel2d_bilapl_tf(x_all_L, x_interior_R))
    # 22 - dx2 with dx2
    K22 = tf.constant(doubledx2_kernel_tf(x_bndry_L, x_bndry_R))
    # 23 - dx2 with dy2
    K23 = tf.constant(dx2_dy2_kernel_tf(x_bndry_L, x_bndry_R))
    # 24 - dx2 with nabla^4 w
    K24 = tf.constant(dx2_bilapl_kernel_tf(x_bndry_L, x_interior_R))
    # 33 - dy2 with dy2
    K33 = tf.constant(doubledy2_kernel_tf(x_bndry_L, x_bndry_R))
    # 34 - dy2 with nabla^4 w
    K34 = tf.constant(dy2_bilapl_kernel_tf(x_bndry_L, x_interior_R))
    # 44 - nabla^4 w with itself
    K44 = tf.constant(kernel2d_double_bilapl_tf(x_interior_L, x_interior_R))

    # print(f"{num_interior=} {num_all=}")
    # print(f"{K11.shape=} {K12.shape=} {K22.shape=}")
    # print(f"{K13.shape=} {K23.shape=} {K33.shape=}")

    # assemble full covariance matrix
    _row1 = tf.constant(tf.concat([K11, K12, K13, K14], axis=1))
    _row2 = tf.constant(tf.concat([tf.transpose(K12), K22, K23, K24], axis=1))
    _row3 = tf.constant(tf.concat([tf.transpose(K13), tf.transpose(K23), K33, K34], axis=1))
    _row4 = tf.constant(tf.concat([tf.transpose(K14), tf.transpose(K24), tf.transpose(K34), K44], axis=1))
    Kblock_prereg = tf.concat([_row1, _row2, _row3, _row4], axis=0)

    # from paper on this by Chen, need to use adaptive nugget term with trace ratio
    tr11 = tf.linalg.trace(K11)
    tr22 = tf.linalg.trace(K22)
    tr33 = tf.linalg.trace(K33)
    tr44 = tf.linalg.trace(K44)
    diagonal = tf.concat([tf.ones((num_all,)), tr22 / tr11 * tf.ones((num_bndry,)), \
                        tr33 / tr11 * tf.ones((num_bndry,)), \
                        tr44 / tr11 * tf.ones((num_interior,))], axis=0)
    regularization = eta * tf.linalg.diag(diagonal)
    plt.imshow(regularization)

    # print("done with eigenvalues solve")

    Kblock = tf.constant(Kblock_prereg + regularization, dtype=DTYPE)

    # # show the matrix image to see if positive definite roughly
    # plt.imshow(Kblock)
    # plt.colorbar()

    Kblock = Kblock.numpy()
    q = q.numpy()

    # newton's algorithm
    # ------------------
    theta = np.zeros((num_interior,1))

    zscale = 1e4

    for inewton in range(2):
        # matrix is dimensions num_interior x num_block
        dzdth = np.concatenate([
            np.eye(num_interior), np.zeros((num_interior,num_interior+3*num_bndry)),
        ], axis=-1) * zscale
        
        # get actual value of z as in nMAP = z^T * Cov^-1 * z
        z = np.concatenate([
            theta, np.zeros((3*num_bndry,1)), q / D
        ], axis=0) * zscale
        # print(f"{z=}")

        # H * dth = -grad is newton update
        temp = np.linalg.solve(Kblock, dzdth.T)
        H = 2 * dzdth @ temp

        temp2 = np.linalg.solve(Kblock, z)
        grad = 2 * dzdth @ temp2

        # newton update
        dtheta = -np.linalg.solve(H, grad)
        # print(f"{dtheta.shape=} {theta.shape=}")
        # theta.assign(theta + dtheta)
        alpha = 1.0
        theta = theta + alpha * dtheta

        # new objective cost
        newz = np.concatenate([
            theta, np.zeros((3*num_bndry,1)), q / D
        ], axis=0) * zscale
        temp3 = np.linalg.solve(Kblock, newz)
        loss = newz.T @ temp3
        # print(f"{inewton=} {loss=}")

    w_full = np.concatenate([
        theta, np.zeros((3*num_bndry,1)), q / D
    ], axis=0)

    # compute the true solution error
    w_pred = w_full[:num_domain,:]
    abs_err = np.max(np.abs((w_pred - w_true_vec).numpy()))
    true_norm = np.max(np.abs(w_true_vec))
    # for now just scale up
    pred_norm = np.max(np.abs(w_pred))
    w_pred2 = w_pred * true_norm/pred_norm
    pred_norm2 = np.max(np.abs(w_pred2))

    abs_err = np.max(np.abs((w_pred2 - w_true_vec).numpy()))

    # print(f"{pred_norm=} {true_norm=} {pred_norm2=}")
    rel_err = abs_err / true_norm
    # print(f"{num_all=}")
    # print(f"{abs_err=} {rel_err=}")

    return num_all, rel_err

if __name__ == "__main__":

    import time
    import matplotlib.pyplot as plt
    import niceplots

    num_interior_0 = 500
    num_bndry_0 = 200
    eta = 1e-9

    num_alls = []
    rel_errs = []
    runtimes = []
    for mesh_scale in [6.0, 8.0]: # [1.0, 1.2]: # 
        for repeat in [1,2,3]:
            start_time = time.time()
            num_all, rel_err = soln_error(eta, int(num_interior_0 * mesh_scale), int(num_bndry_0 * mesh_scale))
            dt = time.time() - start_time

            print(f"{dt=} {num_all=} {rel_err=}", flush=True)

            runtimes += [dt]
            rel_errs += [rel_err]
            num_alls += [num_all]

    print(f"{rel_errs=}")
    print(f"{num_alls=}")
    print(f"{runtimes=}")

    # plt.style.use(niceplots.get_style())
    # plt.plot(num_alls, rel_errs, 'ko')
    # plt.xlabel("Num Collocation Pts")
    # plt.ylabel("Relative Error")
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.show()
    # plt.savefig('lin-static-rel-errors.png', dpi=400)

    # plt.figure('a')
    # plt.plot(num_alls, runtimes, 'ko')
    # plt.xlabel("Num Collocation Pts")
    # plt.ylabel("Runtime (sec)")
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.savefig('lin-static-runtimes.png', dpi=400)
