import time
import matplotlib.pyplot as plt
import niceplots
import numpy as np

plt.style.use(niceplots.get_style())

rel_errs=[0.018337162, 0.09829554, 0.06781079, 0.016071321, 0.025166199, 0.046584535, 0.0041242237, 0.020262085, 0.016324047, 0.019491151, 0.027921632, 0.09301718, 0.031245213, 0.005415314, 0.0032179414]
num_alls=[700, 700, 700, 840, 840, 840, 1050, 1050, 1050, 1400, 1400, 1400, 2100, 2100, 2100]
runtimes=[2.400111675262451, 0.6666064262390137, 0.6780633926391602, 1.1756761074066162, 0.924644947052002, 0.9772019386291504, 1.4130771160125732, 1.413818120956421, 1.4436531066894531, 2.5431759357452393, 2.63871169090271, 2.7355806827545166, 5.337984323501587, 5.141554594039917, 5.149422883987427]

rel_errs+=[0.009802698, 0.001765253, 0.0057929438, 0.01571528, 0.003120096, 0.015440497]
num_alls+=[2800, 2800, 2800, 3500, 3500, 3500]
runtimes+=[12.000516653060913, 10.060849666595459, 9.900675535202026, 15.991029262542725, 15.951586961746216, 15.913296461105347]

# plt.style.use(niceplots.get_style())
plt.plot(num_alls, rel_errs, 'ko')
plt.xlabel("Num Collocation Pts")
plt.ylabel("Relative Error")
plt.xscale('log')
plt.yscale('log')
N = len(rel_errs)
rel_errs_mat = np.reshape(np.array(rel_errs), newshape=(int(N/3),3))
avg_rel_errs = list(np.mean(rel_errs_mat, axis=-1))
plt.plot(num_alls[::3], avg_rel_errs)
plt.margins(x=0.05, y=0.05)

# plt.show()
plt.savefig('lin-static-rel-errors.png', dpi=400)

plt.figure('a')
plt.plot(num_alls, runtimes, 'ko')
plt.xlabel("Num Collocation Pts")
plt.ylabel("Runtime (sec)")
plt.xscale('log')
plt.yscale('log')
plt.savefig('lin-static-runtimes.png', dpi=400)