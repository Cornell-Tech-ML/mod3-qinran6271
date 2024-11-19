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
None
```
Training Result
```
!cd $DIR && PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05
```
```
Epoch  0  loss  6.329380556744396 correct 32 time per epoch 33.36060833930969
Epoch  10  loss  6.428946891384086 correct 34 time per epoch 1.1517863273620605
Epoch  20  loss  7.562574780955919 correct 35 time per epoch 1.140355110168457
Epoch  30  loss  3.931688001519477 correct 49 time per epoch 1.1436259746551514
Epoch  40  loss  3.5463455375286044 correct 49 time per epoch 1.132793664932251
Epoch  50  loss  2.356076034888201 correct 50 time per epoch 1.1409289836883545
Epoch  60  loss  2.52195721232775 correct 50 time per epoch 1.8328328132629395
Epoch  70  loss  2.0129050154614117 correct 50 time per epoch 1.306873083114624
Epoch  80  loss  1.362608633116857 correct 50 time per epoch 1.1439604759216309
Epoch  90  loss  1.8323825079997849 correct 48 time per epoch 1.1459977626800537
Epoch  100  loss  2.555494216200091 correct 49 time per epoch 1.1352922916412354
Epoch  110  loss  0.8235241917061967 correct 50 time per epoch 1.1651153564453125
Epoch  120  loss  0.9058017158579026 correct 50 time per epoch 1.1423373222351074
Epoch  130  loss  1.238432500420562 correct 50 time per epoch 1.1496295928955078
Epoch  140  loss  0.7834865556249283 correct 50 time per epoch 2.0397629737854004
Epoch  150  loss  0.7799926004010437 correct 50 time per epoch 1.1495492458343506
Epoch  160  loss  0.7007451712070379 correct 50 time per epoch 1.1572186946868896
Epoch  170  loss  0.7454541421957442 correct 49 time per epoch 1.1441543102264404
Epoch  180  loss  0.4533742679188853 correct 50 time per epoch 1.135951280593872
Epoch  190  loss  0.5332107073261528 correct 50 time per epoch 1.1635246276855469
Epoch  200  loss  0.2798371952807319 correct 50 time per epoch 1.2976679801940918
Epoch  210  loss  1.0488195889370595 correct 50 time per epoch 1.8361246585845947
Epoch  220  loss  0.3207126464941067 correct 50 time per epoch 1.1420314311981201
Epoch  230  loss  0.24622832614700094 correct 50 time per epoch 1.1365587711334229
Epoch  240  loss  0.5914157838161022 correct 50 time per epoch 1.159560203552246
Epoch  250  loss  0.11849739382688469 correct 50 time per epoch 1.1422479152679443
Epoch  260  loss  0.670686363317095 correct 50 time per epoch 1.15480637550354
Epoch  270  loss  0.6397027925628902 correct 50 time per epoch 1.5931353569030762
Epoch  280  loss  0.14275238353170763 correct 50 time per epoch 1.7088649272918701
Epoch  290  loss  0.4445675038876047 correct 50 time per epoch 1.1428146362304688
Epoch  300  loss  0.35230984899525575 correct 50 time per epoch 1.1370790004730225
Epoch  310  loss  0.1473011178814325 correct 50 time per epoch 1.1463792324066162
Epoch  320  loss  0.1079058693879855 correct 50 time per epoch 1.139085292816162
Epoch  330  loss  0.02048697962613371 correct 50 time per epoch 1.1401588916778564
Epoch  340  loss  0.23431597986258693 correct 50 time per epoch 1.6330485343933105
Epoch  350  loss  0.10887621176770493 correct 50 time per epoch 2.2935988903045654
Epoch  360  loss  0.447113435320587 correct 50 time per epoch 1.1371757984161377
Epoch  370  loss  0.14203008788964 correct 50 time per epoch 1.1411681175231934
Epoch  380  loss  0.32486859928315753 correct 50 time per epoch 1.1715002059936523
Epoch  390  loss  0.07994892870557352 correct 50 time per epoch 1.1511461734771729
Epoch  400  loss  0.030964691452754968 correct 50 time per epoch 1.1501805782318115
Epoch  410  loss  0.10012777257017401 correct 50 time per epoch 1.7415132522583008
Epoch  420  loss  0.14436568400740973 correct 50 time per epoch 1.573392391204834
Epoch  430  loss  0.25200445914814434 correct 50 time per epoch 1.1445648670196533
Epoch  440  loss  0.032494372302225835 correct 50 time per epoch 1.1556003093719482
Epoch  450  loss  0.08271994830560588 correct 50 time per epoch 1.1333134174346924
Epoch  460  loss  0.049489076064449224 correct 50 time per epoch 1.1379351615905762
Epoch  470  loss  0.08828561568200574 correct 50 time per epoch 1.1422865390777588
Epoch  480  loss  0.022156173654027854 correct 50 time per epoch 1.7827162742614746
Epoch  490  loss  0.20661983042631135 correct 50 time per epoch 1.4101402759552002
```
```
python run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05
```
```
Epoch  0  loss  8.838360797088917 correct 38 time per epoch 4.7541725635528564
Epoch  10  loss  4.092803078955783 correct 39 time per epoch 2.6414217948913574
Epoch  20  loss  3.849158967707628 correct 49 time per epoch 1.9597704410552979
Epoch  30  loss  2.3058475341408258 correct 45 time per epoch 1.9619815349578857
Epoch  40  loss  2.466399902154435 correct 46 time per epoch 2.7527406215667725
Epoch  50  loss  1.689306402217929 correct 50 time per epoch 1.9408400058746338
Epoch  60  loss  1.952860668240406 correct 46 time per epoch 2.0255486965179443
Epoch  70  loss  1.5081560687421596 correct 50 time per epoch 2.6275346279144287
Epoch  80  loss  1.0607719130865405 correct 50 time per epoch 1.9796392917633057
Epoch  90  loss  1.0410480030259246 correct 50 time per epoch 2.0627570152282715
Epoch  100  loss  0.48912075438456876 correct 50 time per epoch 2.776263475418091
Epoch  110  loss  0.6515518038095105 correct 50 time per epoch 2.62679386138916
Epoch  120  loss  1.00594056971935 correct 50 time per epoch 1.978724479675293
Epoch  130  loss  1.0942554393077706 correct 50 time per epoch 2.332735776901245
Epoch  140  loss  0.4663971650609539 correct 50 time per epoch 2.0292937755584717
Epoch  150  loss  0.8282750243882028 correct 50 time per epoch 1.9649648666381836
Epoch  160  loss  0.6706867142608739 correct 50 time per epoch 2.240196466445923
Epoch  170  loss  0.4340160340193874 correct 50 time per epoch 1.9701557159423828
Epoch  180  loss  0.5078020753823934 correct 50 time per epoch 2.012385129928589
Epoch  190  loss  0.38466929858387194 correct 50 time per epoch 2.3851871490478516
Epoch  200  loss  0.24677094030376406 correct 50 time per epoch 1.9531254768371582
Epoch  210  loss  0.35362602637914287 correct 50 time per epoch 1.9616243839263916
Epoch  220  loss  0.21003361041103247 correct 50 time per epoch 2.3877813816070557
Epoch  230  loss  0.5887457919628153 correct 50 time per epoch 2.021068811416626
Epoch  240  loss  0.6593639809732923 correct 50 time per epoch 1.9595880508422852
Epoch  250  loss  0.7081668409921991 correct 50 time per epoch 2.7588412761688232
Epoch  260  loss  0.26552814189792606 correct 50 time per epoch 1.9546887874603271
Epoch  270  loss  0.35730925535791846 correct 50 time per epoch 1.9604809284210205
Epoch  280  loss  0.15242408575452954 correct 50 time per epoch 2.8294854164123535
Epoch  290  loss  0.23893001453125468 correct 50 time per epoch 1.9736242294311523
Epoch  300  loss  0.19122161558643383 correct 50 time per epoch 1.9641797542572021
Epoch  310  loss  0.0704185366385677 correct 50 time per epoch 2.783759355545044
Epoch  320  loss  0.1701445034965377 correct 50 time per epoch 2.0212743282318115
Epoch  330  loss  0.1309914564057976 correct 50 time per epoch 2.0356030464172363
Epoch  340  loss  0.31146387691897837 correct 50 time per epoch 3.1662063598632812
Epoch  350  loss  0.32992285513216646 correct 50 time per epoch 1.9661445617675781
Epoch  360  loss  0.23387657785174545 correct 50 time per epoch 1.9599435329437256
Epoch  370  loss  0.22988542049228558 correct 50 time per epoch 2.446458101272583
Epoch  380  loss  0.19275063617624735 correct 50 time per epoch 2.016336441040039
Epoch  390  loss  0.3443280315452528 correct 50 time per epoch 1.9586005210876465
Epoch  400  loss  0.13984485172738298 correct 50 time per epoch 2.2222769260406494
Epoch  410  loss  0.11156510057699474 correct 50 time per epoch 1.9660415649414062
Epoch  420  loss  0.01600530102113003 correct 50 time per epoch 2.0056824684143066
Epoch  430  loss  0.13800487950685197 correct 50 time per epoch 2.1158697605133057
Epoch  440  loss  0.08544419645006626 correct 50 time per epoch 1.9452672004699707
Epoch  450  loss  0.18350165515950667 correct 50 time per epoch 1.9611608982086182
Epoch  460  loss  0.11411525195655844 correct 50 time per epoch 1.9615213871002197
Epoch  470  loss  0.07539741377254147 correct 50 time per epoch 2.0832715034484863
Epoch  480  loss  0.1524078592745861 correct 50 time per epoch 2.400529623031616
Epoch  490  loss  0.08165000781701359 correct 50 time per epoch 1.9694602489471436
```
