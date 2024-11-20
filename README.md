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
# 3.1 and 3.2 parallel check

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
# 3.4 timing

```
{'fast': np.float64(0.033411900202433266), 'gpu': np.float64(0.024042844772338867)}
Running size 256
{'fast': np.float64(0.13466540972391763), 'gpu': np.float64(0.043764750162760414)}
Running size 512
{'fast': np.float64(0.9965393543243408), 'gpu': np.float64(0.18561124801635742)}
Running size 1024
{'fast': np.float64(7.898449818293254), 'gpu': np.float64(0.867194652557373)}

Timing summary
Size: 64
    fast: 0.00764
    gpu: 0.01092
Size: 128
    fast: 0.03341
    gpu: 0.02404
Size: 256
    fast: 0.13467
    gpu: 0.04376
Size: 512
    fast: 0.99654
    gpu: 0.18561
Size: 1024
    fast: 7.89845
    gpu: 0.86719
```
![image](https://github.com/user-attachments/assets/f2d3f337-d54d-4bc6-adfa-911b52a95136)

# 3.5 Training Result
## Regular
### Split
```
python run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05
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
python run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET split --RATE 0.05
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
### Simple
```
python run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET simple --RATE 0.05
```
```
Epoch  0  loss  7.375654965320274 correct 32 time per epoch 33.90684676170349
Epoch  10  loss  2.4334607227378053 correct 48 time per epoch 1.1966617107391357
Epoch  20  loss  1.201746330377227 correct 50 time per epoch 1.1476585865020752
Epoch  30  loss  1.3103624315145186 correct 50 time per epoch 1.397984504699707
Epoch  40  loss  0.680819139529878 correct 50 time per epoch 1.7888765335083008
Epoch  50  loss  1.0135891933431034 correct 50 time per epoch 1.14290452003479
Epoch  60  loss  1.1771956712571163 correct 50 time per epoch 1.1528913974761963
Epoch  70  loss  0.09175955510562298 correct 48 time per epoch 1.143075704574585
Epoch  80  loss  0.475777665757417 correct 50 time per epoch 1.1435153484344482
Epoch  90  loss  0.5161819226034517 correct 50 time per epoch 1.151531457901001
Epoch  100  loss  1.0211878218239203 correct 50 time per epoch 1.4471142292022705
Epoch  110  loss  0.1122068673178497 correct 50 time per epoch 1.8130958080291748
Epoch  120  loss  0.4629174754597052 correct 50 time per epoch 1.1558361053466797
Epoch  130  loss  0.1382419520416368 correct 50 time per epoch 1.1634364128112793
Epoch  140  loss  0.2898245854596878 correct 50 time per epoch 1.1432926654815674
Epoch  150  loss  0.2274803924325247 correct 50 time per epoch 1.1664459705352783
Epoch  160  loss  0.5278782065596894 correct 50 time per epoch 1.1465578079223633
Epoch  170  loss  0.0994600306352594 correct 50 time per epoch 1.43086576461792
Epoch  180  loss  0.26737783930327663 correct 50 time per epoch 1.7233707904815674
Epoch  190  loss  0.08609110692082986 correct 50 time per epoch 1.1494157314300537
Epoch  200  loss  0.1040633196275995 correct 50 time per epoch 1.1553728580474854
Epoch  210  loss  0.05236699356374902 correct 50 time per epoch 1.1509244441986084
Epoch  220  loss  0.5252726275590038 correct 50 time per epoch 1.142265796661377
Epoch  230  loss  0.4151660131191284 correct 50 time per epoch 1.1437458992004395
Epoch  240  loss  0.02331028965281276 correct 50 time per epoch 1.2686281204223633
Epoch  250  loss  0.1795287583518894 correct 50 time per epoch 1.729527473449707
Epoch  260  loss  0.15511962197648094 correct 50 time per epoch 1.8824803829193115
Epoch  270  loss  0.24056439960008855 correct 50 time per epoch 1.1636109352111816
Epoch  280  loss  0.10782862636254287 correct 50 time per epoch 1.1459136009216309
Epoch  290  loss  0.03675771095775766 correct 50 time per epoch 1.1516962051391602
Epoch  300  loss  0.30276042810646087 correct 50 time per epoch 1.1455399990081787
Epoch  310  loss  0.13999098736168705 correct 50 time per epoch 1.1445987224578857
Epoch  320  loss  0.03357051812458975 correct 50 time per epoch 1.3872690200805664
Epoch  330  loss  0.16600679034670357 correct 50 time per epoch 1.7141923904418945
Epoch  340  loss  0.00472938110055548 correct 50 time per epoch 1.1377263069152832
Epoch  350  loss  0.1044359900599182 correct 50 time per epoch 1.1432595252990723
Epoch  360  loss  0.020449139401223226 correct 50 time per epoch 1.1598985195159912
Epoch  370  loss  0.26164803624624455 correct 50 time per epoch 1.154524326324463
Epoch  380  loss  0.05065357640097477 correct 50 time per epoch 1.1558077335357666
Epoch  390  loss  0.09205634401617288 correct 50 time per epoch 1.4877350330352783
Epoch  400  loss  0.11360818505614191 correct 50 time per epoch 1.7245736122131348
Epoch  410  loss  0.038408667091120674 correct 50 time per epoch 1.4229936599731445
Epoch  420  loss  0.14450873323454028 correct 50 time per epoch 1.2678682804107666
Epoch  430  loss  0.21561722394433794 correct 50 time per epoch 1.147279977798462
Epoch  440  loss  0.03216392880082959 correct 50 time per epoch 1.1461379528045654
Epoch  450  loss  0.14984401059895702 correct 50 time per epoch 1.1435110569000244
Epoch  460  loss  0.07404930922998358 correct 50 time per epoch 1.141068696975708
Epoch  470  loss  0.033654897134520224 correct 50 time per epoch 1.6000761985778809
Epoch  480  loss  0.08533199587866386 correct 50 time per epoch 1.6191034317016602
Epoch  490  loss  0.08888240363922231 correct 50 time per epoch 1.1652171611785889
```
```
python run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET simple --RATE 0.05
```
```
Epoch  0  loss  8.48088770180522 correct 24 time per epoch 5.387376070022583
Epoch  10  loss  2.0926607034316014 correct 50 time per epoch 3.081784963607788
Epoch  20  loss  1.6047877867424098 correct 49 time per epoch 1.9714293479919434
Epoch  30  loss  0.7913109166372836 correct 49 time per epoch 1.9768390655517578
Epoch  40  loss  1.1447327418773439 correct 48 time per epoch 2.33744740486145
Epoch  50  loss  0.11326638575815326 correct 49 time per epoch 1.9645943641662598
Epoch  60  loss  0.8282005828171882 correct 50 time per epoch 1.9774994850158691
Epoch  70  loss  0.03161045459824016 correct 50 time per epoch 2.3644468784332275
Epoch  80  loss  1.8123966962053104 correct 50 time per epoch 1.964118242263794
Epoch  90  loss  0.5358712710446599 correct 50 time per epoch 2.030207633972168
Epoch  100  loss  0.03945603516811881 correct 50 time per epoch 2.4420559406280518
Epoch  110  loss  0.8271026648206709 correct 50 time per epoch 1.9730701446533203
Epoch  120  loss  0.9057679761525003 correct 50 time per epoch 2.0053248405456543
Epoch  130  loss  0.5515819455658044 correct 50 time per epoch 2.578050136566162
Epoch  140  loss  0.3107456038616616 correct 50 time per epoch 2.006040334701538
Epoch  150  loss  0.6732173083278971 correct 50 time per epoch 1.967219591140747
Epoch  160  loss  0.8653323786509131 correct 50 time per epoch 2.4012908935546875
Epoch  170  loss  0.44646451527797815 correct 50 time per epoch 2.0020883083343506
Epoch  180  loss  0.17348281693158368 correct 50 time per epoch 2.0339059829711914
Epoch  190  loss  0.7963617012504813 correct 50 time per epoch 2.316364049911499
Epoch  200  loss  0.6546431895760653 correct 50 time per epoch 1.970536470413208
Epoch  210  loss  0.6967445495642687 correct 50 time per epoch 1.965883493423462
Epoch  220  loss  0.13923794514354174 correct 50 time per epoch 1.9630565643310547
Epoch  230  loss  0.04068320082560979 correct 50 time per epoch 2.03364634513855
Epoch  240  loss  1.1901746545703455 correct 50 time per epoch 1.9654834270477295
Epoch  250  loss  0.7345070172178488 correct 50 time per epoch 1.9531948566436768
Epoch  260  loss  0.0009428147125243217 correct 50 time per epoch 1.9607317447662354
Epoch  270  loss  0.4077100093273338 correct 50 time per epoch 1.9931674003601074
Epoch  280  loss  0.2830207002264385 correct 50 time per epoch 2.022951602935791
Epoch  290  loss  0.11648690532508128 correct 50 time per epoch 1.9858460426330566
Epoch  300  loss  0.08454006907558828 correct 50 time per epoch 2.039804220199585
Epoch  310  loss  0.1573607978906545 correct 50 time per epoch 1.9776315689086914
Epoch  320  loss  0.1224338110979502 correct 50 time per epoch 2.473829746246338
Epoch  330  loss  0.7332278876244084 correct 50 time per epoch 2.032066822052002
Epoch  340  loss  0.2609697394662772 correct 50 time per epoch 1.9853076934814453
Epoch  350  loss  0.002205836679387782 correct 50 time per epoch 1.9616098403930664
Epoch  360  loss  0.15255549914030755 correct 50 time per epoch 1.9574265480041504
Epoch  370  loss  0.5810100050146094 correct 50 time per epoch 2.2175681591033936
Epoch  380  loss  0.09398274326853956 correct 50 time per epoch 2.019446849822998
Epoch  390  loss  0.4923759071753301 correct 50 time per epoch 1.9576547145843506
Epoch  400  loss  0.26997420565955227 correct 50 time per epoch 2.3308537006378174
Epoch  410  loss  0.1178292702296782 correct 50 time per epoch 1.9338643550872803
Epoch  420  loss  0.19286760447514661 correct 50 time per epoch 2.01578426361084
Epoch  430  loss  0.5123643284880904 correct 50 time per epoch 2.115100383758545
Epoch  440  loss  0.08012178907122723 correct 50 time per epoch 1.9732985496520996
Epoch  450  loss  0.0074381242193306855 correct 50 time per epoch 1.967078685760498
Epoch  460  loss  0.11485993906788289 correct 50 time per epoch 2.2106845378875732
Epoch  470  loss  0.08702952994268998 correct 50 time per epoch 2.01446533203125
Epoch  480  loss  0.17453114063809472 correct 50 time per epoch 2.039196014404297
Epoch  490  loss  0.40661359691953575 correct 50 time per epoch 2.150908946990967
```
### Xor
```
python run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET xor --RATE 0.05
```
```
Epoch  0  loss  9.402027323473904 correct 23 time per epoch 33.8620982170105
Epoch  10  loss  3.7662757701553806 correct 44 time per epoch 2.093550682067871
Epoch  20  loss  3.7949476806867835 correct 46 time per epoch 1.1264498233795166
Epoch  30  loss  1.983888522194521 correct 43 time per epoch 1.1424376964569092
Epoch  40  loss  2.7691273890424895 correct 47 time per epoch 1.1190977096557617
Epoch  50  loss  3.124008759944757 correct 47 time per epoch 1.1141166687011719
Epoch  60  loss  3.1087515428137067 correct 47 time per epoch 1.1255388259887695
Epoch  70  loss  1.396968286735139 correct 46 time per epoch 2.0188333988189697
Epoch  80  loss  1.4077754476772089 correct 47 time per epoch 1.1195838451385498
Epoch  90  loss  3.853971103418006 correct 46 time per epoch 1.1305744647979736
Epoch  100  loss  2.490360519517406 correct 47 time per epoch 1.1122331619262695
Epoch  110  loss  0.9185136992208198 correct 48 time per epoch 1.1181139945983887
Epoch  120  loss  1.6608272294226432 correct 47 time per epoch 1.1225743293762207
Epoch  130  loss  1.925001908647034 correct 47 time per epoch 1.5454087257385254
Epoch  140  loss  0.4957392878478777 correct 48 time per epoch 1.3571462631225586
Epoch  150  loss  2.72014595722451 correct 48 time per epoch 1.1152727603912354
Epoch  160  loss  3.1300724718741915 correct 49 time per epoch 1.1231021881103516
Epoch  170  loss  1.7639248196025663 correct 48 time per epoch 1.1380441188812256
Epoch  180  loss  2.3982593141713506 correct 48 time per epoch 1.1199109554290771
Epoch  190  loss  1.8620798641244718 correct 49 time per epoch 1.126112937927246
Epoch  200  loss  3.006181626044518 correct 49 time per epoch 1.6404547691345215
Epoch  210  loss  1.0981670433074655 correct 49 time per epoch 1.2070884704589844
Epoch  220  loss  1.2926509148237897 correct 49 time per epoch 1.1252484321594238
Epoch  230  loss  1.8522829012482416 correct 48 time per epoch 1.1255195140838623
Epoch  240  loss  0.30284384226914574 correct 49 time per epoch 1.1240489482879639
Epoch  250  loss  0.8910707488096647 correct 48 time per epoch 1.1326265335083008
Epoch  260  loss  0.3841406133044544 correct 49 time per epoch 1.2807097434997559
Epoch  270  loss  2.4791572108350968 correct 50 time per epoch 2.013296604156494
Epoch  280  loss  0.26386981209581917 correct 50 time per epoch 1.1506028175354004
Epoch  290  loss  1.2675712088575855 correct 50 time per epoch 1.1284127235412598
Epoch  300  loss  1.2156712660252116 correct 50 time per epoch 1.133171796798706
Epoch  310  loss  0.13531567701742345 correct 50 time per epoch 1.1255838871002197
Epoch  320  loss  1.0130898689627172 correct 50 time per epoch 1.1264142990112305
Epoch  330  loss  1.6000290392301966 correct 49 time per epoch 1.5990736484527588
Epoch  340  loss  1.0726514288074631 correct 50 time per epoch 1.3207643032073975
Epoch  350  loss  0.46987158147988095 correct 50 time per epoch 1.1343798637390137
Epoch  360  loss  0.6355604182497379 correct 50 time per epoch 1.1167082786560059
Epoch  370  loss  1.0471682492123124 correct 50 time per epoch 1.122342824935913
Epoch  380  loss  0.5905739627655558 correct 50 time per epoch 1.1442935466766357
Epoch  390  loss  0.684083822642699 correct 50 time per epoch 1.1246705055236816
Epoch  400  loss  0.4484577501975032 correct 50 time per epoch 1.7755043506622314
Epoch  410  loss  0.3318224365622297 correct 50 time per epoch 1.114508867263794
Epoch  420  loss  0.3002621841954713 correct 50 time per epoch 1.2181305885314941
Epoch  430  loss  0.6300730369619958 correct 50 time per epoch 1.136507511138916
Epoch  440  loss  0.579523804056661 correct 50 time per epoch 1.1319103240966797
Epoch  450  loss  0.15720248065499348 correct 50 time per epoch 1.1851346492767334
Epoch  460  loss  0.6448385951402478 correct 50 time per epoch 1.1948215961456299
Epoch  470  loss  0.16187419739528536 correct 50 time per epoch 1.1212153434753418
Epoch  480  loss  0.06337033942101244 correct 50 time per epoch 1.1155729293823242
Epoch  490  loss  0.372465965275979 correct 50 time per epoch 1.5369174480438232
```
```
python run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET xor --RATE 0.05
```
```
Epoch  0  loss  6.503374312539994 correct 29 time per epoch 5.266139984130859
Epoch  10  loss  6.310330712989732 correct 42 time per epoch 2.2944421768188477
Epoch  20  loss  3.404199460507759 correct 44 time per epoch 1.9655656814575195
Epoch  30  loss  6.551660553121677 correct 38 time per epoch 1.987069845199585
Epoch  40  loss  4.305655504558966 correct 46 time per epoch 2.4303672313690186
Epoch  50  loss  3.3879896355021097 correct 45 time per epoch 1.9806268215179443
Epoch  60  loss  4.276422503808437 correct 43 time per epoch 1.9916627407073975
Epoch  70  loss  3.8099569132156423 correct 45 time per epoch 2.7967095375061035
Epoch  80  loss  2.6122512041845822 correct 45 time per epoch 1.9930667877197266
Epoch  90  loss  5.4788289488560515 correct 43 time per epoch 2.0532636642456055
Epoch  100  loss  3.234249322486016 correct 46 time per epoch 2.4774882793426514
Epoch  110  loss  1.6970551121906097 correct 46 time per epoch 1.9715352058410645
Epoch  120  loss  3.455904398719406 correct 45 time per epoch 1.9866862297058105
Epoch  130  loss  2.001943546524232 correct 45 time per epoch 2.305018424987793
Epoch  140  loss  0.6568576526049905 correct 44 time per epoch 2.0212109088897705
Epoch  150  loss  0.9604814102587591 correct 44 time per epoch 2.035255193710327
Epoch  160  loss  1.4658908949032448 correct 48 time per epoch 2.0329151153564453
Epoch  170  loss  2.974783880339385 correct 48 time per epoch 2.3676235675811768
Epoch  180  loss  1.3264548434895589 correct 47 time per epoch 2.0408763885498047
Epoch  190  loss  1.1529603296393869 correct 49 time per epoch 2.052992343902588
Epoch  200  loss  1.6051016405394432 correct 49 time per epoch 2.574861764907837
Epoch  210  loss  1.4579474071068015 correct 49 time per epoch 1.9933929443359375
Epoch  220  loss  2.5076571939548478 correct 45 time per epoch 1.9893102645874023
Epoch  230  loss  0.7976074898159122 correct 50 time per epoch 2.7886147499084473
Epoch  240  loss  0.9809515836471205 correct 49 time per epoch 1.9854679107666016
Epoch  250  loss  1.406380432419411 correct 50 time per epoch 1.9768729209899902
Epoch  260  loss  0.9819559087180311 correct 50 time per epoch 2.6974358558654785
Epoch  270  loss  1.5041950587684747 correct 48 time per epoch 1.9931371212005615
Epoch  280  loss  0.49339030962199093 correct 49 time per epoch 2.035017967224121
Epoch  290  loss  0.8724211868008631 correct 50 time per epoch 2.0870964527130127
Epoch  300  loss  1.1777405969109067 correct 50 time per epoch 1.989121913909912
Epoch  310  loss  1.028460423810265 correct 50 time per epoch 2.0389113426208496
Epoch  320  loss  1.8247256438650785 correct 50 time per epoch 2.0421249866485596
Epoch  330  loss  0.6161068011451456 correct 50 time per epoch 2.1409084796905518
Epoch  340  loss  0.5446656304109252 correct 50 time per epoch 1.9733092784881592
Epoch  350  loss  0.54896332025419 correct 50 time per epoch 2.001793146133423
Epoch  360  loss  1.1634432209720829 correct 50 time per epoch 2.2376301288604736
Epoch  370  loss  1.0038789766264458 correct 50 time per epoch 2.253230094909668
Epoch  380  loss  0.3573788677763482 correct 50 time per epoch 2.0452327728271484
Epoch  390  loss  0.28520939323144656 correct 50 time per epoch 2.843775510787964
Epoch  400  loss  0.15192146145621904 correct 50 time per epoch 1.9924054145812988
Epoch  410  loss  1.210907330232263 correct 50 time per epoch 1.9751255512237549
Epoch  420  loss  0.8754371305409595 correct 50 time per epoch 2.6952953338623047
Epoch  430  loss  0.20397539252976515 correct 49 time per epoch 2.0325965881347656
Epoch  440  loss  0.3759025912843206 correct 50 time per epoch 1.9757628440856934
Epoch  450  loss  0.7898921121973558 correct 50 time per epoch 2.4978532791137695
Epoch  460  loss  0.5619114557445004 correct 50 time per epoch 1.9819445610046387
Epoch  470  loss  0.26350568531420415 correct 50 time per epoch 2.0558059215545654
Epoch  480  loss  0.8241468568156354 correct 50 time per epoch 2.290588617324829
Epoch  490  loss  0.6142915443078133 correct 50 time per epoch 2.017730236053467
```
## Large
### Simple 200 hidden
```
python run_fast_tensor.py --BACKEND cpu --HIDDEN 200 --DATASET split --RATE 0.05
```
```
Epoch  0  loss  3.603934145254928 correct 42 time per epoch 28.91952419281006
Epoch  10  loss  2.0765157849074947 correct 48 time per epoch 4.641663074493408
Epoch  20  loss  0.073783577233813 correct 46 time per epoch 4.843613862991333
Epoch  30  loss  1.338756934910372 correct 49 time per epoch 4.878585577011108
Epoch  40  loss  0.14492232907621472 correct 48 time per epoch 4.877465009689331
Epoch  50  loss  0.07891460912119176 correct 50 time per epoch 4.8995890617370605
Epoch  60  loss  1.4740638434285527 correct 50 time per epoch 4.8377685546875
Epoch  70  loss  0.6672773293958569 correct 50 time per epoch 4.849621295928955
Epoch  80  loss  0.08634911162095607 correct 50 time per epoch 4.541533708572388
Epoch  90  loss  0.1093214255341623 correct 49 time per epoch 4.201634645462036
Epoch  100  loss  0.19653381280599616 correct 48 time per epoch 3.8972327709198
Epoch  110  loss  0.20384177752319188 correct 50 time per epoch 3.9053306579589844
Epoch  120  loss  0.1069662236994094 correct 48 time per epoch 3.898942708969116
Epoch  130  loss  1.2164727100241772 correct 48 time per epoch 3.9047434329986572
Epoch  140  loss  0.29711628591096895 correct 49 time per epoch 3.89286470413208
Epoch  150  loss  0.0388944410930638 correct 49 time per epoch 3.878593921661377
Epoch  160  loss  1.2483563490316685 correct 49 time per epoch 3.8952982425689697
Epoch  170  loss  0.011417129297416048 correct 50 time per epoch 3.8936092853546143
Epoch  180  loss  0.09431116245810531 correct 49 time per epoch 3.9472687244415283
Epoch  190  loss  0.7035570260626398 correct 50 time per epoch 3.8844995498657227
Epoch  200  loss  0.03837635147407693 correct 49 time per epoch 3.8750455379486084
Epoch  210  loss  4.904452560179575 correct 46 time per epoch 3.943676471710205
Epoch  220  loss  0.00407234006978286 correct 50 time per epoch 3.8991024494171143
Epoch  230  loss  0.13174591008388417 correct 49 time per epoch 3.9075615406036377
Epoch  240  loss  0.30113378686845577 correct 50 time per epoch 3.899559259414673
Epoch  250  loss  0.1150161291307482 correct 49 time per epoch 4.146711349487305
Epoch  260  loss  0.6347321176607885 correct 49 time per epoch 4.488407135009766
Epoch  270  loss  0.40114469899179306 correct 50 time per epoch 4.786680459976196
Epoch  280  loss  0.013276310588764448 correct 49 time per epoch 4.870779037475586
Epoch  290  loss  1.3455496443553148 correct 49 time per epoch 4.832920789718628
Epoch  300  loss  0.34238363657339504 correct 50 time per epoch 4.828535795211792
Epoch  310  loss  0.00022617583513073763 correct 49 time per epoch 4.853799343109131
Epoch  320  loss  1.6548998739620449 correct 48 time per epoch 4.824308156967163
Epoch  330  loss  0.17423024140266216 correct 49 time per epoch 4.828672409057617
Epoch  340  loss  0.5182802264704353 correct 49 time per epoch 4.76244330406189
Epoch  350  loss  0.0004352983814297206 correct 49 time per epoch 4.472146272659302
Epoch  360  loss  0.17701214595726975 correct 49 time per epoch 4.160402297973633
Epoch  370  loss  0.23231526322632873 correct 49 time per epoch 3.8734331130981445
Epoch  380  loss  0.002871034895515585 correct 49 time per epoch 3.8793582916259766
Epoch  390  loss  0.10069094943813747 correct 49 time per epoch 3.870987892150879
Epoch  400  loss  0.13176383021945764 correct 49 time per epoch 3.8947207927703857
Epoch  410  loss  0.49613504170790007 correct 49 time per epoch 3.9001755714416504
Epoch  420  loss  0.024701916646848288 correct 49 time per epoch 3.898794174194336
Epoch  430  loss  0.012276403129392443 correct 50 time per epoch 3.8833463191986084
Epoch  440  loss  0.832985454353329 correct 49 time per epoch 3.9259140491485596
Epoch  450  loss  0.03699215615579663 correct 49 time per epoch 3.887876033782959
Epoch  460  loss  1.293801539147706 correct 49 time per epoch 3.8945906162261963
Epoch  470  loss  0.07427238665692387 correct 49 time per epoch 3.89217209815979
Epoch  480  loss  0.02102479188071152 correct 49 time per epoch 3.9086716175079346
Epoch  490  loss  1.2225679528880466 correct 49 time per epoch 3.9266645908355713
```
```
python run_fast_tensor.py --BACKEND gpu --HIDDEN 200 --DATASET split --RATE 0.05
```
```
Epoch  0  loss  2.0688989144411005 correct 44 time per epoch 6.111421823501587
Epoch  10  loss  1.65740287148855 correct 49 time per epoch 3.03564715385437
Epoch  20  loss  0.4031533166463632 correct 49 time per epoch 2.9541616439819336
Epoch  30  loss  0.45807991899612577 correct 49 time per epoch 2.90478777885437
Epoch  40  loss  0.13057641028193462 correct 50 time per epoch 3.6451263427734375
Epoch  50  loss  0.32177653237936127 correct 48 time per epoch 2.911586046218872
Epoch  60  loss  0.2985046761385746 correct 50 time per epoch 3.2746968269348145
Epoch  70  loss  0.22764831887253198 correct 50 time per epoch 2.886296033859253
Epoch  80  loss  0.04352825714723398 correct 50 time per epoch 3.000781774520874
Epoch  90  loss  0.05872193277571272 correct 50 time per epoch 3.82383131980896
Epoch  100  loss  0.032242510511538085 correct 50 time per epoch 2.906853199005127
Epoch  110  loss  0.005425265123886717 correct 50 time per epoch 3.120673656463623
Epoch  120  loss  0.539108557447397 correct 50 time per epoch 2.9395081996917725
Epoch  130  loss  0.08188408480754047 correct 50 time per epoch 3.014373779296875
Epoch  140  loss  0.17123671919578942 correct 50 time per epoch 3.739428758621216
Epoch  150  loss  0.014372778271424351 correct 50 time per epoch 2.9245517253875732
Epoch  160  loss  0.01794405600140453 correct 50 time per epoch 3.2553958892822266
Epoch  170  loss  0.008347383020092677 correct 50 time per epoch 3.014652967453003
Epoch  180  loss  0.17102741892790996 correct 50 time per epoch 2.984203815460205
Epoch  190  loss  0.006631432712300384 correct 50 time per epoch 3.6390786170959473
Epoch  200  loss  0.4546728218713001 correct 50 time per epoch 2.93188214302063
Epoch  210  loss  0.015261902459081395 correct 50 time per epoch 3.495054006576538
Epoch  220  loss  0.11834220227027341 correct 50 time per epoch 2.9364211559295654
Epoch  230  loss  0.09653491984775087 correct 50 time per epoch 3.0089917182922363
Epoch  240  loss  0.07285158130910036 correct 50 time per epoch 2.9648799896240234
Epoch  250  loss  0.7273351402992554 correct 50 time per epoch 2.949760913848877
Epoch  260  loss  0.32098754298849647 correct 50 time per epoch 3.80395245552063
Epoch  270  loss  0.06706738337285643 correct 50 time per epoch 2.9370620250701904
Epoch  280  loss  0.24069268940372943 correct 50 time per epoch 3.3595473766326904
Epoch  290  loss  0.187790954777176 correct 50 time per epoch 2.9099371433258057
Epoch  300  loss  0.002349419464551348 correct 50 time per epoch 2.893902540206909
Epoch  310  loss  0.00011408226953358205 correct 50 time per epoch 3.7620651721954346
Epoch  320  loss  0.09632217262785589 correct 50 time per epoch 2.9821078777313232
Epoch  330  loss  0.24815409683675135 correct 50 time per epoch 3.1435415744781494
Epoch  340  loss  0.0008680237584740001 correct 50 time per epoch 2.898132085800171
Epoch  350  loss  0.31050022837128166 correct 50 time per epoch 2.9295151233673096
Epoch  360  loss  0.21169267342693265 correct 50 time per epoch 3.751490831375122
Epoch  370  loss  -6.39879671511241e-06 correct 50 time per epoch 2.972954034805298
Epoch  380  loss  0.1995355671519968 correct 50 time per epoch 3.1135053634643555
Epoch  390  loss  0.012906939889593506 correct 50 time per epoch 2.9432976245880127
Epoch  400  loss  0.012608276866375346 correct 50 time per epoch 2.8927652835845947
Epoch  410  loss  0.06952244698084466 correct 50 time per epoch 3.731564998626709
Epoch  420  loss  0.037690127333528355 correct 50 time per epoch 2.956852674484253
Epoch  430  loss  0.13338800482270619 correct 50 time per epoch 2.9597153663635254
Epoch  450  loss  0.046646699877471234 correct 50 time per epoch 2.892881155014038
Epoch  460  loss  1.2216792046237106e-05 correct 50 time per epoch 3.6517386436462402
Epoch  470  loss  0.03538433514719154 correct 50 time per epoch 2.9867985248565674
Epoch  480  loss  0.15082012347003704 correct 50 time per epoch 2.982696294784546
Epoch  490  loss  0.15174153641864943 correct 50 time per epoch 2.9165408611297607
```
