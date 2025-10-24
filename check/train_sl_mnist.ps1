python launch.py `
--model 'SL_MLP' `
--task 'MNIST' `
--archi 784 512 10 `
--alg 'EP' `
--optim 'sgd' `
--epochs 20 `
--act 'identity' `
--todo 'train' `
--thirdphase `
--input-positive-negative-mapping `
--loss 'complex_mse' `
--betas 0.1 0.1 `
--epsilon 0.05 `
--T1 30 `
--T2 10 `
--mbs 256 `
--device 0 `
--multi-gpu `
--cache-to-gpu `
--save  

return

python launch.py `
--model 'SL_MLP' `
--task 'CIFAR10' `
--archi 784 512 10 `
--alg 'EP' `
--optim 'sgd' `
--epochs 20 `
--act 'identity' `
--todo 'train' `
--thirdphase `
--input-positive-negative-mapping `
--loss 'complex_mse' `
--betas 0.1 0.1 `
--epsilon 0.05 `
--T1 30 `
--T2 10 `
--mbs 64 `
--device 0 `
--num-repeats 5 `
--num-workers 5 `
--save  

python launch.py `
--model 'OIM_MLP' `
--task 'MNIST' `
--archi 784 512 10 `
--alg 'cos' `
--optim 'sgd' `
--epochs 20 `
--act '' `
--todo 'train' `
--thirdphase `
--input-positive-negative-mapping `
--loss 'mse' `
--betas 0.1 0.1 `
--epsilon 0.05 `
--T1 30 `
--T2 10 `
--mbs 64 `
--device 0 `
--num-repeats 10 `
--num-workers 10 `
--save  

python launch.py `
--model 'OIM_MLP' `
--task 'CIFAR10' `
--archi 784 512 10 `
--alg 'cos' `
--optim 'sgd' `
--epochs 20 `
--act 'identity' `
--todo 'train' `
--thirdphase `
--input-positive-negative-mapping `
--loss 'mse' `
--betas 0.1 0.1 `
--epsilon 0.05 `
--T1 30 `
--T2 10 `
--mbs 64 `
--device 0 `
--num-repeats 10 `
--num-workers 10 `
--save  

python launch.py `
--model 'P_MLP' `
--task 'MNIST' `
--archi 784 512 10 `
--alg 'BPTT' `
--optim 'sgd' `
--thirdphase `
--epochs 20 `
--act 'tanh' `
--todo 'train' `
--input-positive-negative-mapping `
--loss 'mse' `
--betas 0.1 0.1 `
--epsilon 0.05 `
--T1 30 `
--T2 10 `
--mbs 64 `
--device 0 `
--num-repeats 10 `
--num-workers 10 `
--save  

python launch.py `
--model 'P_MLP' `
--task 'CIFAR10' `
--archi 784 512 10 `
--alg 'BPTT' `
--optim 'sgd' `
--thirdphase `
--epochs 20 `
--act 'tanh' `
--todo 'train' `
--input-positive-negative-mapping `
--loss 'mse' `
--betas 0.1 0.1 `
--epsilon 0.05 `
--T1 30 `
--T2 10 `
--mbs 64 `
--device 0 `
--num-repeats 10 `
--num-workers 10 `
--save  

