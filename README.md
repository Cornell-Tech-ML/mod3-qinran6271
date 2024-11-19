# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py
Result of python project/parallel_check.py

```
MAP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, /Users/qin
ranwang/Downloads/Cornell/MLE/MiniTorchWorkspace/mod3-
qinran6271/minitorch/fast_ops.py (164)  
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/qinranwang/Downloads/Cornell/MLE/MiniTorchWorkspace/mod3-qinran6271/minitorch/fast_ops.py (164) 
----------------------------------------------------------------------------------|loop #ID
    def _map(                                                                     | 
        out: Storage,                                                             | 
        out_shape: Shape,                                                         | 
        out_strides: Strides,                                                     | 
        in_storage: Storage,                                                      | 
        in_shape: Shape,                                                          | 
        in_strides: Strides,                                                      | 
    ) -> None:                                                                    | 
        # TODO: Implement for Task 3.1.                                           | 
                                                                                  | 
        out_index2d: Index = np.zeros((2, len(out), MAX_DIMS), dtype=np.int32)----| #0
                                                                                  | 
        if np.array_equal(in_strides, out_strides) and np.array_equal(            | 
            in_shape, out_shape                                                   | 
        ):                                                                        | 
            for i in prange(len(out)):--------------------------------------------| #1
                out[i] = fn(in_storage[i])                                        | 
        else:                                                                     | 
            for i in prange(len(out)):--------------------------------------------| #2
                out_index: Index = out_index2d[0, i]                              | 
                in_index: Index = out_index2d[1, i]                               | 
                to_index(i, out_shape, out_index)                                 | 
                broadcast_index(out_index, out_shape, in_shape, in_index)         | 
                o = index_to_position(out_index, out_strides)                     | 
                j = index_to_position(in_index, in_strides)                       | 
                out[o] = fn(in_storage[j])                                        | 
            # raise NotImplementedError("Need to implement for Task 3.1")         | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #0, #1, #2).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
```
```
ZIP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, /Users/qin
ranwang/Downloads/Cornell/MLE/MiniTorchWorkspace/mod3-
qinran6271/minitorch/fast_ops.py (218)  
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/qinranwang/Downloads/Cornell/MLE/MiniTorchWorkspace/mod3-qinran6271/minitorch/fast_ops.py (218) 
----------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                     | 
        out: Storage,                                                             | 
        out_shape: Shape,                                                         | 
        out_strides: Strides,                                                     | 
        a_storage: Storage,                                                       | 
        a_shape: Shape,                                                           | 
        a_strides: Strides,                                                       | 
        b_storage: Storage,                                                       | 
        b_shape: Shape,                                                           | 
        b_strides: Strides,                                                       | 
    ) -> None:                                                                    | 
        # TODO: Implement for Task 3.1.                                           | 
        out_index2d: Index = np.zeros((3, len(out), MAX_DIMS), dtype=np.int32)----| #3
        if (                                                                      | 
            np.array_equal(out_strides, a_strides)                                | 
            and np.array_equal(out_strides, b_strides)                            | 
            and np.array_equal(out_shape, a_shape)                                | 
            and np.array_equal(out_shape, b_shape)                                | 
        ):                                                                        | 
            for i in prange(len(out)):--------------------------------------------| #4
                out[i] = fn(a_storage[i], b_storage[i])                           | 
        else:                                                                     | 
            for i in prange(len(out)):--------------------------------------------| #5
                # out_index: Index = np.zeros(MAX_DIMS, np.int32)                 | 
                # a_index: Index = np.zeros(MAX_DIMS, np.int32)                   | 
                # b_index: Index = np.zeros(MAX_DIMS, np.int32)                   | 
                out_index: Index = out_index2d[0, i]                              | 
                a_index: Index = out_index2d[1, i]                                | 
                b_index: Index = out_index2d[2, i]                                | 
                to_index(i, out_shape, out_index)                                 | 
                o = index_to_position(out_index, out_strides)                     | 
                broadcast_index(out_index, out_shape, a_shape, a_index)           | 
                j = index_to_position(a_index, a_strides)                         | 
                broadcast_index(out_index, out_shape, b_shape, b_index)           | 
                k = index_to_position(b_index, b_strides)                         | 
                out[o] = fn(a_storage[j], b_storage[k])                           | 
        # raise NotImplementedError("Need to implement for Task 3.1")             | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #3, #4, #5).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
```
```
REDUCE
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, /Use
rs/qinranwang/Downloads/Cornell/MLE/MiniTorchWorkspace/mod3-
qinran6271/minitorch/fast_ops.py (280)  
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/qinranwang/Downloads/Cornell/MLE/MiniTorchWorkspace/mod3-qinran6271/minitorch/fast_ops.py (280) 
----------------------------------------------------------------------------------|loop #ID
    def _reduce(                                                                  | 
        out: Storage,                                                             | 
        out_shape: Shape,                                                         | 
        out_strides: Strides,                                                     | 
        a_storage: Storage,                                                       | 
        a_shape: Shape,                                                           | 
        a_strides: Strides,                                                       | 
        reduce_dim: int,                                                          | 
    ) -> None:                                                                    | 
        # TODO: Implement for Task 3.1.                                           | 
        out_index2d: Index = np.zeros((1, len(out), MAX_DIMS), dtype=np.int32)----| #6
        for i in prange(len(out)):------------------------------------------------| #7
            out_index: Index = out_index2d[0, i]                                  | 
            # out_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)               | 
            reduce_size = a_shape[reduce_dim]                                     | 
            to_index(i, out_shape, out_index)                                     | 
            o = index_to_position(out_index, out_strides)                         | 
                                                                                  | 
            # reduce across the reduce_dim                                        | 
            j = index_to_position(out_index, a_strides)                           | 
            acc = out[o]                                                          | 
            step = a_strides[reduce_dim]                                          | 
            for _ in range(reduce_size):                                          | 
                acc = fn(acc, a_storage[j])                                       | 
                j += step                                                         | 
            out[o] = acc                                                          | 
        # raise NotImplementedError("Need to implement for Task 3.1")             | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #6, #7).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
```
```
MATRIX MULTIPLY
 
================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, /Users/qinr
anwang/Downloads/Cornell/MLE/MiniTorchWorkspace/mod3-
qinran6271/minitorch/fast_ops.py (311)  
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/qinranwang/Downloads/Cornell/MLE/MiniTorchWorkspace/mod3-qinran6271/minitorch/fast_ops.py (311) 
---------------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                                       | 
    out: Storage,                                                                                  | 
    out_shape: Shape,                                                                              | 
    out_strides: Strides,                                                                          | 
    a_storage: Storage,                                                                            | 
    a_shape: Shape,                                                                                | 
    a_strides: Strides,                                                                            | 
    b_storage: Storage,                                                                            | 
    b_shape: Shape,                                                                                | 
    b_strides: Strides,                                                                            | 
) -> None:                                                                                         | 
    """NUMBA tensor matrix multiply function.                                                      | 
                                                                                                   | 
    Should work for any tensor shapes that broadcast as long as                                    | 
                                                                                                   | 
    ```                                                                                            | 
    assert a_shape[-1] == b_shape[-2]                                                              | 
    ```                                                                                            | 
                                                                                                   | 
    Optimizations:                                                                                 | 
                                                                                                   | 
    * Outer loop in parallel                                                                       | 
    * No index buffers or function calls                                                           | 
    * Inner loop should have no global writes, 1 multiply.                                         | 
                                                                                                   | 
                                                                                                   | 
    Args:                                                                                          | 
    ----                                                                                           | 
        out (Storage): storage for `out` tensor                                                    | 
        out_shape (Shape): shape for `out` tensor                                                  | 
        out_strides (Strides): strides for `out` tensor                                            | 
        a_storage (Storage): storage for `a` tensor                                                | 
        a_shape (Shape): shape for `a` tensor                                                      | 
        a_strides (Strides): strides for `a` tensor                                                | 
        b_storage (Storage): storage for `b` tensor                                                | 
        b_shape (Shape): shape for `b` tensor                                                      | 
        b_strides (Strides): strides for `b` tensor                                                | 
                                                                                                   | 
    Returns:                                                                                       | 
    -------                                                                                        | 
        None : Fills in `out`                                                                      | 
                                                                                                   | 
    """                                                                                            | 
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                         | 
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                         | 
                                                                                                   | 
    # TODO: Implement for Task 3.2.                                                                | 
                                                                                                   | 
    # raise NotImplementedError("Need to implement for Task 3.2")                                  | 
    assert a_shape[-1] == b_shape[-2]                                                              | 
    # Outer loop over output tensor in parallel                                                    | 
    # Outer loop over output tensor in parallel                                                    | 
    for i in prange(out_shape[0]):  # Assuming batch dimension (or outermost dimension)------------| #8
        for j in range(out_shape[1]):  # Iterate over rows of output                               | 
            for k in range(out_shape[2]):  # Iterate over columns of output                        | 
                # a_index = i * a_batch_stride + j * a_strides[1] + p * a_strides[2] # 第j行， 第p列    | 
                a_base_index = i * a_batch_stride + j * a_strides[1]                               | 
                # b_index = i * b_batch_stride + p * b_strides[1] + k * b_strides[2] # 第p行， 第k列    | 
                b_base_index = i * b_batch_stride + k * b_strides[2]                               | 
                                                                                                   | 
                # Initialize output                                                                | 
                out_index = i * out_strides[0] + j * out_strides[1] + k * out_strides[2]           | 
                out_value = 0.0  # Use local variable to avoid repeated writes                     | 
                                                                                                   | 
                # Compute dot product along the common dimension                                   | 
                for p in range(a_shape[-1]):                                                       | 
                    out_value += (                                                                 | 
                        a_storage[a_base_index + p * a_strides[2]]                                 | 
                        * b_storage[b_base_index + p * b_strides[1]]                               | 
                    )                                                                              | 
                                                                                                   | 
                # Store the result in the output tensor                                            | 
                out[out_index] = out_value                                                         | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #8).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None```
```!cd $DIR && PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05
```
```Epoch  0  loss  6.4968888464704735 correct 36 time per epoch 24.144663333892822
Epoch  10  loss  3.9578409459282784 correct 38 time per epoch 0.9791371822357178
Epoch  20  loss  4.679231531783756 correct 45 time per epoch 0.9093866348266602
Epoch  30  loss  2.533831123665224 correct 40 time per epoch 0.9338433742523193
Epoch  40  loss  3.370185502995285 correct 46 time per epoch 1.4380383491516113
Epoch  50  loss  3.418897575982477 correct 45 time per epoch 0.9135429859161377
Epoch  60  loss  2.2016947390878348 correct 48 time per epoch 0.9403231143951416
Epoch  70  loss  2.8747559385576658 correct 48 time per epoch 0.9452097415924072
Epoch  80  loss  1.9113285166076417 correct 48 time per epoch 0.9434607028961182
Epoch  90  loss  3.1300700139547724 correct 48 time per epoch 0.9334080219268799
Epoch  100  loss  1.207114035020312 correct 48 time per epoch 0.9140996932983398
Epoch  110  loss  1.9806607219327277 correct 49 time per epoch 1.480525255203247
Epoch  120  loss  1.2213321864305187 correct 48 time per epoch 0.9121599197387695
Epoch  130  loss  0.8221558875794364 correct 48 time per epoch 0.9173905849456787
Epoch  140  loss  3.2082374479454687 correct 50 time per epoch 1.228849172592163
Epoch  150  loss  2.330796540192317 correct 48 time per epoch 0.9231059551239014
Epoch  160  loss  0.7301903505671454 correct 48 time per epoch 0.9355053901672363
Epoch  170  loss  0.17809215066991088 correct 49 time per epoch 0.9608933925628662
Epoch  180  loss  1.107191216184606 correct 49 time per epoch 1.3284339904785156
Epoch  190  loss  1.4522752446561809 correct 50 time per epoch 1.0004668235778809
Epoch  200  loss  2.8305166127836854 correct 50 time per epoch 0.9976966381072998
Epoch  210  loss  0.3936421168271946 correct 48 time per epoch 0.9774825572967529
Epoch  220  loss  0.43468324009249365 correct 49 time per epoch 1.757211685180664
Epoch  230  loss  1.3651859154072148 correct 48 time per epoch 0.9911048412322998
Epoch  240  loss  0.663207283857236 correct 49 time per epoch 1.0275843143463135
Epoch  250  loss  0.5222191999306071 correct 48 time per epoch 1.0559890270233154
Epoch  260  loss  0.5024639062846329 correct 50 time per epoch 1.3073523044586182
Epoch  270  loss  1.6029907858346635 correct 48 time per epoch 0.9902005195617676
Epoch  280  loss  0.7305996732546078 correct 50 time per epoch 0.9909260272979736
Epoch  290  loss  0.4139446694576468 correct 49 time per epoch 0.9816999435424805
Epoch  300  loss  0.6080005465995468 correct 49 time per epoch 1.022411584854126
Epoch  310  loss  0.3362258451054042 correct 49 time per epoch 1.5687050819396973
Epoch  320  loss  0.14184513148773703 correct 50 time per epoch 0.9895312786102295
Epoch  330  loss  0.9439022942731701 correct 49 time per epoch 0.9714310169219971
Epoch  340  loss  0.6161865596032075 correct 49 time per epoch 1.0730631351470947
Epoch  350  loss  0.9655329935660806 correct 49 time per epoch 1.606121301651001
Epoch  360  loss  2.0436519618772886 correct 49 time per epoch 1.0233376026153564
Epoch  370  loss  0.5702782289754218 correct 50 time per epoch 1.0463120937347412
Epoch  380  loss  0.17781147479647794 correct 49 time per epoch 1.0227060317993164
Epoch  390  loss  2.1852808058823614 correct 49 time per epoch 1.0293357372283936
Epoch  400  loss  0.24148062174387291 correct 50 time per epoch 1.3931610584259033
Epoch  410  loss  1.0569607578771532 correct 49 time per epoch 1.027207851409912
Epoch  420  loss  0.45968000663288305 correct 49 time per epoch 1.0328338146209717
Epoch  430  loss  0.727441667945834 correct 50 time per epoch 0.9941041469573975
Epoch  440  loss  1.7136714825713428 correct 50 time per epoch 1.8374199867248535
Epoch  450  loss  1.056346736968349 correct 50 time per epoch 1.031541109085083
Epoch  460  loss  0.12896124422291394 correct 49 time per epoch 1.0312836170196533
Epoch  470  loss  0.5917835124482559 correct 49 time per epoch 0.9789412021636963
Epoch  480  loss  1.086330054164532 correct 50 time per epoch 1.1061632633209229
Epoch  490  loss  0.6031218030801195 correct 49 time per epoch 1.036116600036621```
