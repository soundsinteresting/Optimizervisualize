# RMSprop
This repository is the official implementation of the paper "RMSprop can converge with proper hyper-parameter".

This folder contains the following code: 

 (a) cifar_resnet.py: this is the code for RMSprop/Adam algorithm, for training cifar10 on resnet, presented in Section 5;
 
 (b) cifar_resnet_SGD.py: this is the code for SGD algorithm with momentum, for training cifar10 on resnet, presented in Section 5;
 
 (c) reddiexample.m: this is the code for Adam algorithm for training the counter-example (1) in Reddi et. al, presented in Section 1;
 
 (d) JSGAN-code folder: this is the code for running RMSprop algorithm, for training cifar10 on JS-GAN, presented in Appendix A.1.
 
 (e) realizable.m: this is the code for running Adam algorithm for the realizable example (6), presented in Appendix A.2;
 
 (f) nonrealizable.m: this is the code for running RMSprop/Adam algorithm for the nonrealizable example, presented in Appendix A.3.
 
 (g) Losssurface folder: this is the code for plotting the loss surface and the trajectories of RMSprop with large or small \beta2.

These scripts do not depend on other scripts, and can be run on its own. They only require standard packages like PyTorch, as specified at the beginning of the code.

The results and figures in the paper are obtained by the following:

 (1) Figure 1 is obtained by directly running reddiexample.m in the default setting.
 
 (2) Table 2 and Table 3 are obtained by running cifar_renet.py and cifar_renet_SGD.py. Specifically, we run each of the setting (with different beta2 and batchsize) 3 times and calculate the mean and standard deviation. The learning rate is fixed as 1e-3 in this experiment.
 
 (3) Table 4 is obtained by running JSGAN code. Specifically, we run each of the setting (with different beta2) 3 times and calculate the mean and standard deviation. For details of running JSGAN code, see readme_gan.txt.
 
 (4) Figure A2 is obtained by running realizable.m. The default setting is beta2=0.999, which yields Figure A2(d). When changing beta2, other figures are obtained.
 
 (5) Figure A3 is obtained by running nonrealizable.m. The default setting is RMSprop beta2=0.999, which yields Figure A3(a). When changing the algorithm to AMSgrad or SGD, other figures are obtained.
 
 (6) Figure 2 is obtained by running the loss surface experiment. For details, see instruction.txt.
