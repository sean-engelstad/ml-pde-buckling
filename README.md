Architectures that use the strong form
1 - PINN uses strong form with static sampling of the domain.
2 - DGM (Deep Galerkin Machine) also uses strong form with static sampling of the domain.

Architectures that use the weak form:
1 - Deep Ritz method uses the energy functional of BVP (boundary value problems). Guess I kind of already tried this one.
2 - Variational PINNs, incorporate the weak form into loss function by predicting test functions for the network.
3 - Finite element inspired neural nets
4 - Professors max-min approach, see notes
5 - Look at the other approaches for eigenvalue problems in the literature and the papers in the proposal

Would like to try different types of networks as well:
1 - Standard FNN (fully connected neural net)
2 - Fourier Neural Operator (FNN in Fourier space)
3 - others?

Operator learning as well:
1 - Deep ONet, etc.

Also: other main comments:
* how to achieve deeper solution? see multi-stage NN paper which achieves machine precision solutions.
* how to solve simple problems faster when you know they have low wavenumbers
* can we constrain the wavenumbers in the model?
* can we solve multiple eigenmodes by orthogonality constraint and repeating the domain inputs => phi outputs in parallel?
* can FNO (Fourier neural operator) help us constrain the wavenumbers?
* feels wrong that it takes us so long to converge to very simple low wavenumber solutions. has to be cleaner way.

* Can I somehow add a constraint to FNO or regular FNN that the max wavenumber is less than the quad point spacing (so numerical integral is valid)?

Also: would like to try Gaussian process method to solve PDEs and look at that book for better FEM basis functions with Gaussian processes
* Gaussian process methods
* wavelets and other operator methods
Look at this paper by Chen, "Solving and learning nonlinear PDEs with Gaussian processes"