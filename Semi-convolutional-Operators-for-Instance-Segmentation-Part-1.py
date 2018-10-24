
# coding: utf-8

# # Semi-convolutional Operators for Instance Segmentation - Part 1

# ### Instance segmentation through pixel colouring
# - Let $\mathbf{x}$ be an image with $\geq 1$ with various objects that we wish to segment individually
# - We want to be able to differentiate not only between objects of different categories but different instances of objects from the same category within a single image
# - One approach is to map each pixel to a continuous real number or "colour" given by $\Phi_u(\mathbf{x})$ for a pixel $u$.
# - The colour values must be close for all pairs pixels within the same instance, parameterised by a margin $M$:
# 
#     $$\lVert \Phi_u(\mathbf{x}) - \Phi_v(\mathbf{x})\rVert^2 \leq 1 - M$$
# 
# - They should be further apart for all pairs of pixels from different instances, 
# 
#     $$\lVert \Phi_u(\mathbf{x}) - \Phi_v(\mathbf{x})\rVert^2 \geq 1 + M, $$
# 
# - The problem with using convolution networks for this purpose is that they are translation invariant.
# - So if there are replicas of an object in the image the network will assign the same colour to each. 

# ### Semi-convolutions
# - We take the output of a convolutional operator and mix it with the pixel location information using a function $f$ to obtain a non-convolutional response  $\Psi_u(\mathbf{x}) =  f(\Phi_u(\mathbf{x}), u)$
# - A simple way of realising this is to make $f$ the addition operator so that $ \Psi_u(\mathbf{x}) =  \Phi_u(\mathbf{x}) +  u$
#     
# 
# - For two different pixels $u,v, u \neq v$ within a instance it must be that $\Psi_u(\mathbf{x}) = \Psi_v(\mathbf{x})$ 
# 
# - Thus for all the pixels $u$ in an instance $S_k$, it must be that $\Psi_u(\mathbf{x}) = \Phi_u(\mathbf{x}) + u = c_k$
#     
# - We can think of $c_k$ as the centroid of the instance.
# - In this interpretation the convolutional operator outputs the displacement of each pixel in the instance from the centroid.

# ### Steered bilateral kernels
# - Almost always two instances of the same object will have some distinctive traits.
# - To incorporate these we allow $\Phi_u(\mathbf{x})$ to have additional dimensions.
# - But eventually we still need to recover location information at the end.
# - To so consider the Gaussian kernel
# 
# $$K(u, v) = \exp\left(-\frac{\lVert\Psi_u(\mathbf{x}) - \Psi_v(\mathbf{x})\rVert^2}{2}\right)$$
# 
# - Suppose that $\Phi_u(\mathbf{x})$ is made to be $d$ dimensional (with $u$ padded with $d-2$ zeros for the addition step).
# - Then squared difference term becomes
# 
#     $$\lVert\Phi_u(\mathbf{x}) - \Psi_v(\mathbf{x})\rVert^2
#         = \sum_i\left(\Psi_{u,i}(\mathbf{x}) - \Psi_{v,i}(\mathbf{x})\right)^2
#         \\ = \sum_{i=1}^{2}\left((\Phi_{u,i}(\mathbf{x}) - u_i)  - (\Phi_{v,i}(\mathbf{x}) - v_i)\right)^2
#         + \sum_{i=3}^{d}\left(\Phi_{u,i}(\mathbf{x})  - \Phi_{v,i}(\mathbf{x})\right)^2$$
# 
# - Thus the kernel can be expressed as the product of a geometric part and an appearance part
# 
# $$K(u, v) = \exp\left(-\frac{\lVert(\Phi_u^g(\mathbf{x}) - u)  - (\Phi_v^g(\mathbf{x}) - v)\rVert^2}{2}\right)\exp\left(-\frac{\lVert\Phi_u^a(\mathbf{x})  - \Phi_v^a(\mathbf{x})\rVert^2}{2}\right)$$
# 
# - A more common similar formulation called the *bilateral kernel*
# 
# $$K(u, v) = \exp\left(-\frac{\lVert u - v\rVert^2}{2}\right)\exp\left(-\frac{\lVert\Phi_u^a(\mathbf{x})  - \Phi_v^a(\mathbf{x})\rVert^2}{2}\right)$$
# 
# - The difference here is the absence of the displacements added by $\Phi_u^g(\mathbf{x})$ and $\Phi_v^g(\mathbf{x})$.
# - So we can think of the new formulation as a 'steered' bilateral kernel where the pixel locations have been distorted by the network to bring points within an instance together.
