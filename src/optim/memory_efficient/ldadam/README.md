# LDAdam

Torch implementation of the LDAdam algorithm. For further details regarding the algorithm, please refer to: [LDAdam: Adaptive Optimization from Low-Dimensional Gradient Statistics](https://arxiv.org/abs/2410.16103).

**Parameters**

   * **params** (iterable) -  iterable of parameters to optimize or dicts defining parameter groups
   * **lr** ([float](https://docs.python.org/3/library/functions.html#float), optional) – learning rate. Default: 1e-3
   * **betas** (Tuple[[float](https://docs.python.org/3/library/functions.html#float), [float](https://docs.python.org/3/library/functions.html#float)], optional) – coefficients used for computing running averages of gradient and its square. Default: (0.908, 0.99)
   * **eps** ([float](https://docs.python.org/3/library/functions.html#float), optional) – term added to the denominator to improve numerical stability. Default: 1e-8
   * **weight_decay** ([float](https://docs.python.org/3/library/functions.html#float), optional) – weight decay (L2 penalty). Default: 0.0
   * **rank** ([int](https://docs.python.org/3/library/functions.html#int), optional) - low-rank compression rank. Default: 16
   * **rho** ([float](https://docs.python.org/3/library/functions.html#float), optional) - interpolation factor. Default: 0.908
   * **proj_type** ([str](https://docs.python.org/3/library/stdtypes.html#str), optional) - low-rank projection type : 'std' | 'left' | 'right' | 'reverse_std'. Default: 'std'
   * **proj_method** ([str](https://docs.python.org/3/library/stdtypes.html#str), optional) - method used for SVD computation: 'power_iteration' | 'svd' | 'svd_lowrank'. Default: 'power_iteration'
   * **no_error_feedback** ([bool](https://docs.python.org/3/library/functions.html#bool), optional) - whether generalized error feedback mechanism is not used. Default: False