# Human Pose Estimation

![Model Image](/Images/hrnet_final.png)

* Features:
  1. Bottleneck Layers
  2. Basic Block
  3. Transition Layers
      * Used for creating the sub branch of the current branch having the resolution 1/2 of the previous branch.
      * There are total 3 transitions in the model so the resolutions are of the order x(original), x/2, x/4, x/8.
  4. Fuse Layers
      * Used for fusing the previous stage module to the next stage module.
      * Each stage module has its own fuse layer and the output of the fuse layer has the same no. of modules as the previous it only changes after the transition.
      * The fuse layer creates the resultant module by either passing the block, downsampling the higher to the lower block, upsampling the lower block to the higher block and finally concatenating the blocks in such a way that they have same resoltion for a particular part in the block and finally ends with ReLU activation.
      * The final fuse layer just upsamples the lower final blocks of the branches and concatenates with the uppermost final block of the branch thus having the maximum resolution from total no. of branches.
      * The type of upsampling used is nearest upsampling method.
      ![Fuse Image](/Images/Fuse.PNG)
      * For more details on upsampling visit https://pytorch.org/docs/stable/nn.html#upsample.
