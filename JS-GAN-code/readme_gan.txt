This is the code for running the JSGAN experiment in appendix A.1. In the experiment, the default learning rate is set to be 2e-4 for both generator and discriminator. The number of iterations is 100000. 3 repetitions are run for each beta2.

Details of the code:

vanillaGAN.py: train JS-GAN
datasets.py: get dataloader
networks.py, disc_resblocksbnorm.py, gen_resblocks.py: define networks structure
eval.py, fid_tf.py, inception_score_tf.py, prd_score.py: evaluate GANs wrt FID, IS and Precision-Recall

Example to run DCGAN on CIFAR-10: 
   JS-GAN: python vanillaGAN.py --dataset cifar --batch_size 64 --Dnum_features 64 --Gnum_features 64  
