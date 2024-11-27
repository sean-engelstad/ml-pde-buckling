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

