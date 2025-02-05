This directory provides a script to run convolution benchmarks on shapes provided in MIOpen format
The expected shape format looks like below
```
Count	Driver command
25	./bin/MIOpenDriver convbfp16 -n 128 -c 128 -H 24 -W 48 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
17	./bin/MIOpenDriver convbfp16 -n 128 -c 128 -H 24 -W 48 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
4	./bin/MIOpenDriver convbfp16 -n 128 -c 35 -H 48 -W 32 -k 35 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
3	./bin/MIOpenDriver convbfp16 -n 128 -c 35 -H 48 -W 32 -k 35 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1
49	./bin/MIOpenDriver convbfp16 -n 128 -c 384 -H 24 -W 48 -k 384 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 1 -t 1
```

Here is what the arguments mean from MIOpenDriver
```
MIOpenDriver conv --help
MIOpen Driver Input Flags:

      --in_d               -!        Input Depth (Default=32)
      --conv_stride_d      -#        Convolution Stride for Depth (Default=1)
      --pad_d              -$        Zero Padding for Depth (Default=0)
      --trans_output_pad_d -%        Zero Padding Output for Depth (Default=0)
      --fil_d              -@        Filter Depth (Default=3)
      --verification_cache -C        Use specified directory to cache verification data. Off by default.
      --dout_data          -D        dy data filename for backward weight computation (Default=)
      --forw               -F        Flag enables fwd, bwd, wrw convolutions
                                     0 fwd+bwd+wrw (default)
                                     1 fwd only
                                     2 bwd only
                                     4 wrw only
                                     3 fwd+bwd
                                     5 fwd+wrw
                                     6 bwd+wrw
      --gpualloc           -G        Controls allocation and initialization buffers on GPU and CPU.
                                     0 Init input buffers on CPU and copy them to GPU. After convolution
                                       is executed, copy output buffer to CPU (Default).
                                     1 No copying. Use hipMalloc to allocate and rocrand to init buffers
                                       directly on GPU. Verification (-V 1) won't succeed in this mode.
      --in_h               -H        Input Height (Default=32)
      --in_layout          -I        Input Layout (Default=NCHW for 2d conv, NCDHW for 3d conv)
      --vector_length      -L        tensor vectorization length (Default=1)
      --out_layout         -O        Output Layout (Default=NCHW for 2d conv, NCDHW for 3d conv)
      --printconv          -P        Print Convolution Dimensions (Default=1)
      --wei_cast_type      -R        Cast type for weight tensor, default to not set
      --solution           -S        Use immediate mode, run solution with specified id.
                                     Accepts integer argument N:
                                     =0 Immediate mode, build and run fastest solution
                                     >0 Immediate mode, build and run solution_id = N
                                     <0 Use Find() API (Default=-1)
                                     Also accepts symbolic name of solution:
                                     <valid name>   Immediate mode, build and run specified solution
                                     <invalid name> Use Find() API
      --out_cast_type      -T        Cast type for output tensor, default to not set
      --in_cast_type       -U        Cast type for input tensor, default to not set
      --verify             -V        Verify Each Layer (Default=1)
      --in_w               -W        Input Width (Default=32)
      --trans_output_pad_w -X        Zero Padding Output for Width (Default=0)
      --trans_output_pad_h -Y        Zero Padding Output for Height (Default=0)
      --tensor_vect        -Z        tensor vectorization type (none, vect_c, vect_n) (Default=0)
      --dilation_d         -^        Dilation of Filter Depth (Default=1)
      --spatial_dim        -_        convolution spatial dimension (Default-2)
      --in_bias            -a        Input bias filename (Default=)
      --bias               -b        Use Bias (Default=0)
      --in_channels        -c        Number of Input Channels (Default=3)
      --in_data            -d        Input data filename (Default=)
      --weights            -e        Input weights filename (Default=)
      --fil_layout         -f        Filter Layout (Default=NCHW for 2d conv, NCDHW for 3d conv)
      --group_count        -g        Number of Groups (Default=1)
      --help               -h        Print Help Message
      --iter               -i        Number of Iterations (Default=10)
      --dilation_w         -j        Dilation of Filter Width (Default=1)
      --out_channels       -k        Number of Output Channels (Default=32)
      --dilation_h         -l        Dilation of Filter Height (Default=1)
      --mode               -m        Convolution Mode (conv, trans) (Default=conv)
      --batchsize          -n        Mini-batch size (Default=100)
      --dump_output        -o        Dumps the output buffers (Default=0)
      --pad_h              -p        Zero Padding for Height (Default=0)
      --pad_w              -q        Zero Padding for Width (Default=0)
      --pad_val            -r        Padding Value (Default=0)
      --search             -s        Search Kernel Config (Default=0)
      --time               -t        Time Each Layer (Default=0)
      --conv_stride_h      -u        Convolution Stride for Height (Default=1)
      --conv_stride_w      -v        Convolution Stride for Width (Default=1)
      --wall               -w        Wall-clock Time Each Layer
                                     0 Off (Default)
                                     1 On, requires '--time 1')
                                     2 On, warm-up the library (prefetch db caches), requires '--time 1'
      --fil_w              -x        Filter Width (Default=3)
      --fil_h              -y        Filter Height (Default=3)
      --pad_mode           -z        Padding Mode (same, valid, default) (Default=default)
```
# Here are the steps to get IREE numbers
Note that a lof of these arguments are simply ignored, but cases such
as non-unit dialations, strides, group convs where IREE GPU backend wont use IGEMM.
Additionally user of the script is expected to only pass 2D convs to the script
for now.

## Step 1
Copy the shapes to a text file without the header. e.g shapes.txt

## Step 2
export PYTHONPATH=~/iree-kernel-benchmark/convbench:$PYTHONPATH

## step 3
python miopendriver_to_iree.py shapes.txt

## step 4
The supported shapes and the perf numbers will be populated in miopen_results.csv