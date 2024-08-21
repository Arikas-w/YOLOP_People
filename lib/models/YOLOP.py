
import sys,os
import sys
sys.path.append(os.getcwd())
from lib.config import cfg

"""
MCnet_SPP = [
[ -1, Focus, [3, 32, 3]],
[ -1, Conv, [32, 64, 3, 2]],
[ -1, BottleneckCSP, [64, 64, 1]],
[ -1, Conv, [64, 128, 3, 2]],
[ -1, BottleneckCSP, [128, 128, 3]],
[ -1, Conv, [128, 256, 3, 2]],
[ -1, BottleneckCSP, [256, 256, 3]],
[ -1, Conv, [256, 512, 3, 2]],
[ -1, SPP, [512, 512, [5, 9, 13]]],
[ -1, BottleneckCSP, [512, 512, 1, False]],
[ -1, Conv,[512, 256, 1, 1]],
[ -1, Upsample, [None, 2, 'nearest']],
[ [-1, 6], Concat, [1]],
[ -1, BottleneckCSP, [512, 256, 1, False]],
[ -1, Conv, [256, 128, 1, 1]],
[ -1, Upsample, [None, 2, 'nearest']],
[ [-1,4], Concat, [1]],
[ -1, BottleneckCSP, [256, 128, 1, False]],
[ -1, Conv, [128, 128, 3, 2]],
[ [-1, 14], Concat, [1]],
[ -1, BottleneckCSP, [256, 256, 1, False]],
[ -1, Conv, [256, 256, 3, 2]],
[ [-1, 10], Concat, [1]],
[ -1, BottleneckCSP, [512, 512, 1, False]],
# [ [17, 20, 23], Detect,  [1, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512]]],
[ [17, 20, 23], Detect,  [13, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512]]],
[ 17, Conv, [128, 64, 3, 1]],
[ -1, Upsample, [None, 2, 'nearest']],
[ [-1,2], Concat, [1]],
[ -1, BottleneckCSP, [128, 64, 1, False]],
[ -1, Conv, [64, 32, 3, 1]],
[ -1, Upsample, [None, 2, 'nearest']],
[ -1, Conv, [32, 16, 3, 1]],
[ -1, BottleneckCSP, [16, 8, 1, False]],
[ -1, Upsample, [None, 2, 'nearest']],
[ -1, SPP, [8, 2, [5, 9, 13]]] #segmentation output
]
# [2,6,3,9,5,13], [7,19,11,26,17,39], [28,64,44,103,61,183]

MCnet_0 = [
[ -1, Focus, [3, 32, 3]],
[ -1, Conv, [32, 64, 3, 2]],
[ -1, BottleneckCSP, [64, 64, 1]],
[ -1, Conv, [64, 128, 3, 2]],
[ -1, BottleneckCSP, [128, 128, 3]],
[ -1, Conv, [128, 256, 3, 2]],
[ -1, BottleneckCSP, [256, 256, 3]],
[ -1, Conv, [256, 512, 3, 2]],
[ -1, SPP, [512, 512, [5, 9, 13]]],
[ -1, BottleneckCSP, [512, 512, 1, False]],
[ -1, Conv,[512, 256, 1, 1]],
[ -1, Upsample, [None, 2, 'nearest']],
[ [-1, 6], Concat, [1]],
[ -1, BottleneckCSP, [512, 256, 1, False]],
[ -1, Conv, [256, 128, 1, 1]],
[ -1, Upsample, [None, 2, 'nearest']],
[ [-1,4], Concat, [1]],
[ -1, BottleneckCSP, [256, 128, 1, False]],
[ -1, Conv, [128, 128, 3, 2]],
[ [-1, 14], Concat, [1]],
[ -1, BottleneckCSP, [256, 256, 1, False]],
[ -1, Conv, [256, 256, 3, 2]],
[ [-1, 10], Concat, [1]],
[ -1, BottleneckCSP, [512, 512, 1, False]],
[ [17, 20, 23], Detect,  [1, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512]]], #Detect output 24

[ 16, Conv, [128, 64, 3, 1]],
[ -1, Upsample, [None, 2, 'nearest']],
[ [-1,2], Concat, [1]],
[ -1, BottleneckCSP, [128, 64, 1, False]],
[ -1, Conv, [64, 32, 3, 1]],
[ -1, Upsample, [None, 2, 'nearest']],
[ -1, Conv, [32, 16, 3, 1]],
[ -1, BottleneckCSP, [16, 8, 1, False]],
[ -1, Upsample, [None, 2, 'nearest']],
[ -1, Conv, [8, 2, 3, 1]], #Driving area segmentation output

[ 16, Conv, [128, 64, 3, 1]],
[ -1, Upsample, [None, 2, 'nearest']],
[ [-1,2], Concat, [1]],
[ -1, BottleneckCSP, [128, 64, 1, False]],
[ -1, Conv, [64, 32, 3, 1]],
[ -1, Upsample, [None, 2, 'nearest']],
[ -1, Conv, [32, 16, 3, 1]],
[ -1, BottleneckCSP, [16, 8, 1, False]],
[ -1, Upsample, [None, 2, 'nearest']],
[ -1, Conv, [8, 2, 3, 1]], #Lane line segmentation output
]


# The lane line and the driving area segment branches share information with each other
MCnet_share = [
[ -1, Focus, [3, 32, 3]],   #0
[ -1, Conv, [32, 64, 3, 2]],    #1
[ -1, BottleneckCSP, [64, 64, 1]],  #2
[ -1, Conv, [64, 128, 3, 2]],   #3
[ -1, BottleneckCSP, [128, 128, 3]],    #4
[ -1, Conv, [128, 256, 3, 2]],  #5
[ -1, BottleneckCSP, [256, 256, 3]],    #6
[ -1, Conv, [256, 512, 3, 2]],  #7
[ -1, SPP, [512, 512, [5, 9, 13]]],     #8
[ -1, BottleneckCSP, [512, 512, 1, False]],     #9
[ -1, Conv,[512, 256, 1, 1]],   #10
[ -1, Upsample, [None, 2, 'nearest']],  #11
[ [-1, 6], Concat, [1]],    #12
[ -1, BottleneckCSP, [512, 256, 1, False]], #13
[ -1, Conv, [256, 128, 1, 1]],  #14
[ -1, Upsample, [None, 2, 'nearest']],  #15
[ [-1,4], Concat, [1]],     #16
[ -1, BottleneckCSP, [256, 128, 1, False]],     #17
[ -1, Conv, [128, 128, 3, 2]],      #18
[ [-1, 14], Concat, [1]],       #19
[ -1, BottleneckCSP, [256, 256, 1, False]],     #20
[ -1, Conv, [256, 256, 3, 2]],      #21
[ [-1, 10], Concat, [1]],   #22
[ -1, BottleneckCSP, [512, 512, 1, False]],     #23
[ [17, 20, 23], Detect,  [1, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512]]], #Detect output 24

[ 16, Conv, [256, 64, 3, 1]],   #25
[ -1, Upsample, [None, 2, 'nearest']],  #26
[ [-1,2], Concat, [1]],  #27
[ -1, BottleneckCSP, [128, 64, 1, False]],  #28
[ -1, Conv, [64, 32, 3, 1]],    #29
[ -1, Upsample, [None, 2, 'nearest']],  #30
[ -1, Conv, [32, 16, 3, 1]],    #31
[ -1, BottleneckCSP, [16, 8, 1, False]],    #32 driving area segment neck

[ 16, Conv, [256, 64, 3, 1]],   #33
[ -1, Upsample, [None, 2, 'nearest']],  #34
[ [-1,2], Concat, [1]], #35
[ -1, BottleneckCSP, [128, 64, 1, False]],  #36
[ -1, Conv, [64, 32, 3, 1]],    #37
[ -1, Upsample, [None, 2, 'nearest']],  #38
[ -1, Conv, [32, 16, 3, 1]],    #39   
[ -1, BottleneckCSP, [16, 8, 1, False]],    #40 lane line segment neck

[ [31,39], Concat, [1]],    #41
[ -1, Conv, [32, 8, 3, 1]],     #42    Share_Block


[ [32,42], Concat, [1]],     #43
[ -1, Upsample, [None, 2, 'nearest']],  #44
[ -1, Conv, [16, 2, 3, 1]], #45 Driving area segmentation output


[ [40,42], Concat, [1]],    #46
[ -1, Upsample, [None, 2, 'nearest']],  #47
[ -1, Conv, [16, 2, 3, 1]] #48Lane line segmentation output
]

# The lane line and the driving area segment branches without share information with each other
MCnet_no_share = [
[ -1, Focus, [3, 32, 3]],   #0
[ -1, Conv, [32, 64, 3, 2]],    #1
[ -1, BottleneckCSP, [64, 64, 1]],  #2
[ -1, Conv, [64, 128, 3, 2]],   #3
[ -1, BottleneckCSP, [128, 128, 3]],    #4
[ -1, Conv, [128, 256, 3, 2]],  #5
[ -1, BottleneckCSP, [256, 256, 3]],    #6
[ -1, Conv, [256, 512, 3, 2]],  #7
[ -1, SPP, [512, 512, [5, 9, 13]]],     #8
[ -1, BottleneckCSP, [512, 512, 1, False]],     #9
[ -1, Conv,[512, 256, 1, 1]],   #10
[ -1, Upsample, [None, 2, 'nearest']],  #11
[ [-1, 6], Concat, [1]],    #12
[ -1, BottleneckCSP, [512, 256, 1, False]], #13
[ -1, Conv, [256, 128, 1, 1]],  #14
[ -1, Upsample, [None, 2, 'nearest']],  #15
[ [-1,4], Concat, [1]],     #16
[ -1, BottleneckCSP, [256, 128, 1, False]],     #17
[ -1, Conv, [128, 128, 3, 2]],      #18
[ [-1, 14], Concat, [1]],       #19
[ -1, BottleneckCSP, [256, 256, 1, False]],     #20
[ -1, Conv, [256, 256, 3, 2]],      #21
[ [-1, 10], Concat, [1]],   #22
[ -1, BottleneckCSP, [512, 512, 1, False]],     #23
[ [17, 20, 23], Detect,  [1, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512]]], #Detect output 24

[ 16, Conv, [256, 64, 3, 1]],   #25
[ -1, Upsample, [None, 2, 'nearest']],  #26
[ [-1,2], Concat, [1]],  #27
[ -1, BottleneckCSP, [128, 64, 1, False]],  #28
[ -1, Conv, [64, 32, 3, 1]],    #29
[ -1, Upsample, [None, 2, 'nearest']],  #30
[ -1, Conv, [32, 16, 3, 1]],    #31
[ -1, BottleneckCSP, [16, 8, 1, False]],    #32 driving area segment neck
[ -1, Upsample, [None, 2, 'nearest']],  #33
[ -1, Conv, [8, 3, 3, 1]], #34 Driving area segmentation output

[ 16, Conv, [256, 64, 3, 1]],   #35
[ -1, Upsample, [None, 2, 'nearest']],  #36
[ [-1,2], Concat, [1]], #37
[ -1, BottleneckCSP, [128, 64, 1, False]],  #38
[ -1, Conv, [64, 32, 3, 1]],    #39
[ -1, Upsample, [None, 2, 'nearest']],  #40
[ -1, Conv, [32, 16, 3, 1]],    #41
[ -1, BottleneckCSP, [16, 8, 1, False]],    #42 lane line segment neck
[ -1, Upsample, [None, 2, 'nearest']],  #43
[ -1, Conv, [8, 2, 3, 1]] #44 Lane line segmentation output
]

MCnet_feedback = [
[ -1, Focus, [3, 32, 3]],   #0
[ -1, Conv, [32, 64, 3, 2]],    #1
[ -1, BottleneckCSP, [64, 64, 1]],  #2
[ -1, Conv, [64, 128, 3, 2]],   #3
[ -1, BottleneckCSP, [128, 128, 3]],    #4
[ -1, Conv, [128, 256, 3, 2]],  #5
[ -1, BottleneckCSP, [256, 256, 3]],    #6
[ -1, Conv, [256, 512, 3, 2]],  #7
[ -1, SPP, [512, 512, [5, 9, 13]]],     #8
[ -1, BottleneckCSP, [512, 512, 1, False]],     #9
[ -1, Conv,[512, 256, 1, 1]],   #10
[ -1, Upsample, [None, 2, 'nearest']],  #11
[ [-1, 6], Concat, [1]],    #12
[ -1, BottleneckCSP, [512, 256, 1, False]], #13
[ -1, Conv, [256, 128, 1, 1]],  #14
[ -1, Upsample, [None, 2, 'nearest']],  #15
[ [-1,4], Concat, [1]],     #16
[ -1, BottleneckCSP, [256, 128, 1, False]],     #17
[ -1, Conv, [128, 128, 3, 2]],      #18
[ [-1, 14], Concat, [1]],       #19
[ -1, BottleneckCSP, [256, 256, 1, False]],     #20
[ -1, Conv, [256, 256, 3, 2]],      #21
[ [-1, 10], Concat, [1]],   #22
[ -1, BottleneckCSP, [512, 512, 1, False]],     #23
[ [17, 20, 23], Detect,  [1, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512]]], #Detect output 24

[ 16, Conv, [256, 128, 3, 1]],   #25
[ -1, Upsample, [None, 2, 'nearest']],  #26
[ -1, BottleneckCSP, [128, 64, 1, False]],  #28
[ -1, Conv, [64, 32, 3, 1]],    #29
[ -1, Upsample, [None, 2, 'nearest']],  #30
[ -1, Conv, [32, 16, 3, 1]],    #31
[ -1, BottleneckCSP, [16, 8, 1, False]],    #32 driving area segment neck
[ -1, Upsample, [None, 2, 'nearest']],  #33
[ -1, Conv, [8, 2, 3, 1]], #34 Driving area segmentation output

[ 16, Conv, [256, 128, 3, 1]],   #35
[ -1, Upsample, [None, 2, 'nearest']],  #36
[ -1, BottleneckCSP, [128, 64, 1, False]],  #38
[ -1, Conv, [64, 32, 3, 1]],    #39
[ -1, Upsample, [None, 2, 'nearest']],  #40
[ -1, Conv, [32, 16, 3, 1]],    #41
[ -1, BottleneckCSP, [16, 8, 1, False]],    #42 lane line segment neck
[ -1, Upsample, [None, 2, 'nearest']],  #43
[ -1, Conv, [8, 2, 3, 1]] #44 Lane line segmentation output
]


MCnet_Da_feedback1 = [
[46, 26, 35],   #Det_out_idx, Da_Segout_idx, LL_Segout_idx
[ -1, Focus, [3, 32, 3]],   #0
[ -1, Conv, [32, 64, 3, 2]],    #1
[ -1, BottleneckCSP, [64, 64, 1]],  #2
[ -1, Conv, [64, 128, 3, 2]],   #3
[ -1, BottleneckCSP, [128, 128, 3]],    #4
[ -1, Conv, [128, 256, 3, 2]],  #5
[ -1, BottleneckCSP, [256, 256, 3]],    #6
[ -1, Conv, [256, 512, 3, 2]],  #7
[ -1, SPP, [512, 512, [5, 9, 13]]],     #8
[ -1, BottleneckCSP, [512, 512, 1, False]],     #9
[ -1, Conv,[512, 256, 1, 1]],   #10
[ -1, Upsample, [None, 2, 'nearest']],  #11
[ [-1, 6], Concat, [1]],    #12
[ -1, BottleneckCSP, [512, 256, 1, False]], #13
[ -1, Conv, [256, 128, 1, 1]],  #14
[ -1, Upsample, [None, 2, 'nearest']],  #15
[ [-1,4], Concat, [1]],     #16     backbone+fpn
[ -1,Conv,[256,256,1,1]],   #17


[ 16, Conv, [256, 128, 3, 1]],   #18
[ -1, Upsample, [None, 2, 'nearest']],  #19
[ -1, BottleneckCSP, [128, 64, 1, False]],  #20
[ -1, Conv, [64, 32, 3, 1]],    #21
[ -1, Upsample, [None, 2, 'nearest']],  #22
[ -1, Conv, [32, 16, 3, 1]],    #23
[ -1, BottleneckCSP, [16, 8, 1, False]],    #24 driving area segment neck
[ -1, Upsample, [None, 2, 'nearest']],  #25
[ -1, Conv, [8, 2, 3, 1]], #26 Driving area segmentation output


[ 16, Conv, [256, 128, 3, 1]],   #27
[ -1, Upsample, [None, 2, 'nearest']],  #28
[ -1, BottleneckCSP, [128, 64, 1, False]],  #29
[ -1, Conv, [64, 32, 3, 1]],    #30
[ -1, Upsample, [None, 2, 'nearest']],  #31
[ -1, Conv, [32, 16, 3, 1]],    #32
[ -1, BottleneckCSP, [16, 8, 1, False]],    #33 lane line segment neck
[ -1, Upsample, [None, 2, 'nearest']],  #34
[ -1, Conv, [8, 2, 3, 1]], #35Lane line segmentation output


[ 23, Conv, [16, 16, 3, 2]],     #36
[ -1, Conv, [16, 32, 3, 2]],    #2 times 2xdownsample    37

[ [-1,17], Concat, [1]],       #38
[ -1, BottleneckCSP, [288, 128, 1, False]],    #39
[ -1, Conv, [128, 128, 3, 2]],      #40
[ [-1, 14], Concat, [1]],       #41
[ -1, BottleneckCSP, [256, 256, 1, False]],     #42
[ -1, Conv, [256, 256, 3, 2]],      #43
[ [-1, 10], Concat, [1]],   #44
[ -1, BottleneckCSP, [512, 512, 1, False]],     #45
[ [39, 42, 45], Detect,  [1, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512]]] #Detect output 46
]



# The lane line and the driving area segment branches share information with each other and feedback to det_head
MCnet_Da_feedback2 = [
[47, 26, 35],   #Det_out_idx, Da_Segout_idx, LL_Segout_idx
[25, 28, 31, 33],   #layer in Da_branch to do SAD
[34, 37, 40, 42],   #layer in LL_branch to do SAD
[ -1, Focus, [3, 32, 3]],   #0
[ -1, Conv, [32, 64, 3, 2]],    #1
[ -1, BottleneckCSP, [64, 64, 1]],  #2
[ -1, Conv, [64, 128, 3, 2]],   #3
[ -1, BottleneckCSP, [128, 128, 3]],    #4
[ -1, Conv, [128, 256, 3, 2]],  #5
[ -1, BottleneckCSP, [256, 256, 3]],    #6
[ -1, Conv, [256, 512, 3, 2]],  #7
[ -1, SPP, [512, 512, [5, 9, 13]]],     #8
[ -1, BottleneckCSP, [512, 512, 1, False]],     #9
[ -1, Conv,[512, 256, 1, 1]],   #10
[ -1, Upsample, [None, 2, 'nearest']],  #11
[ [-1, 6], Concat, [1]],    #12
[ -1, BottleneckCSP, [512, 256, 1, False]], #13
[ -1, Conv, [256, 128, 1, 1]],  #14
[ -1, Upsample, [None, 2, 'nearest']],  #15
[ [-1,4], Concat, [1]],     #16     backbone+fpn
[ -1,Conv,[256,256,1,1]],   #17


[ 16, Conv, [256, 128, 3, 1]],   #18
[ -1, Upsample, [None, 2, 'nearest']],  #19
[ -1, BottleneckCSP, [128, 64, 1, False]],  #20
[ -1, Conv, [64, 32, 3, 1]],    #21
[ -1, Upsample, [None, 2, 'nearest']],  #22
[ -1, Conv, [32, 16, 3, 1]],    #23
[ -1, BottleneckCSP, [16, 8, 1, False]],    #24 driving area segment neck
[ -1, Upsample, [None, 2, 'nearest']],  #25
[ -1, Conv, [8, 2, 3, 1]], #26 Driving area segmentation output


[ 16, Conv, [256, 128, 3, 1]],   #27
[ -1, Upsample, [None, 2, 'nearest']],  #28
[ -1, BottleneckCSP, [128, 64, 1, False]],  #29
[ -1, Conv, [64, 32, 3, 1]],    #30
[ -1, Upsample, [None, 2, 'nearest']],  #31
[ -1, Conv, [32, 16, 3, 1]],    #32
[ -1, BottleneckCSP, [16, 8, 1, False]],    #33 lane line segment neck
[ -1, Upsample, [None, 2, 'nearest']],  #34
[ -1, Conv, [8, 2, 3, 1]], #35Lane line segmentation output


[ 23, Conv, [16, 64, 3, 2]],     #36
[ -1, Conv, [64, 256, 3, 2]],    #2 times 2xdownsample    37

[ [-1,17], Concat, [1]],       #38

[-1, Conv, [512, 256, 3, 1]],     #39
[ -1, BottleneckCSP, [256, 128, 1, False]],    #40
[ -1, Conv, [128, 128, 3, 2]],      #41
[ [-1, 14], Concat, [1]],       #42
[ -1, BottleneckCSP, [256, 256, 1, False]],     #43
[ -1, Conv, [256, 256, 3, 2]],      #44
[ [-1, 10], Concat, [1]],   #45
[ -1, BottleneckCSP, [512, 512, 1, False]],     #46
[ [40, 42, 45], Detect,  [1, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512]]] #Detect output 47
]

MCnet_share1 = [
[24, 33, 45],   #Det_out_idx, Da_Segout_idx, LL_Segout_idx
[25, 28, 31, 33],   #layer in Da_branch to do SAD
[34, 37, 40, 42],   #layer in LL_branch to do SAD
[ -1, Focus, [3, 32, 3]],   #0
[ -1, Conv, [32, 64, 3, 2]],    #1
[ -1, BottleneckCSP, [64, 64, 1]],  #2
[ -1, Conv, [64, 128, 3, 2]],   #3
[ -1, BottleneckCSP, [128, 128, 3]],    #4
[ -1, Conv, [128, 256, 3, 2]],  #5
[ -1, BottleneckCSP, [256, 256, 3]],    #6
[ -1, Conv, [256, 512, 3, 2]],  #7
[ -1, SPP, [512, 512, [5, 9, 13]]],     #8
[ -1, BottleneckCSP, [512, 512, 1, False]],     #9
[ -1, Conv,[512, 256, 1, 1]],   #10
[ -1, Upsample, [None, 2, 'nearest']],  #11
[ [-1, 6], Concat, [1]],    #12
[ -1, BottleneckCSP, [512, 256, 1, False]], #13
[ -1, Conv, [256, 128, 1, 1]],  #14
[ -1, Upsample, [None, 2, 'nearest']],  #15
[ [-1,4], Concat, [1]],     #16
[ -1, BottleneckCSP, [256, 128, 1, False]],     #17
[ -1, Conv, [128, 128, 3, 2]],      #18
[ [-1, 14], Concat, [1]],       #19
[ -1, BottleneckCSP, [256, 256, 1, False]],     #20
[ -1, Conv, [256, 256, 3, 2]],      #21
[ [-1, 10], Concat, [1]],   #22
[ -1, BottleneckCSP, [512, 512, 1, False]],     #23
[ [17, 20, 23], Detect,  [1, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512]]], #Detect output 24

[ 16, Conv, [256, 128, 3, 1]],   #25
[ -1, Upsample, [None, 2, 'nearest']],  #26
[ -1, BottleneckCSP, [128, 64, 1, False]],  #27
[ -1, Conv, [64, 32, 3, 1]],    #28
[ -1, Upsample, [None, 2, 'nearest']],  #29
[ -1, Conv, [32, 16, 3, 1]],    #30

[ -1, BottleneckCSP, [16, 8, 1, False]],    #31 driving area segment neck
[ -1, Upsample, [None, 2, 'nearest']],  #32
[ -1, Conv, [8, 2, 3, 1]], #33 Driving area segmentation output

[ 16, Conv, [256, 128, 3, 1]],   #34
[ -1, Upsample, [None, 2, 'nearest']],  #35
[ -1, BottleneckCSP, [128, 64, 1, False]],  #36
[ -1, Conv, [64, 32, 3, 1]],    #37
[ -1, Upsample, [None, 2, 'nearest']],  #38
[ -1, Conv, [32, 16, 3, 1]],    #39

[ 30, SharpenConv, [16,16, 3, 1]], #40
[ -1, Conv, [16, 16, 3, 1]], #41
[ [-1, 39], Concat, [1]],   #42
[ -1, BottleneckCSP, [32, 8, 1, False]],    #43 lane line segment neck
[ -1, Upsample, [None, 2, 'nearest']],  #44
[ -1, Conv, [8, 2, 3, 1]] #45 Lane line segmentation output
]"""


# The lane line and the driving area segment branches without share information with each other and without link
YOLOP = [
[24, 33, 42],   #Det_out_idx, Da_Segout_idx, LL_Segout_idx
[ -1, 'Focus', [3, 32, 3]],   #0   1/2 ch_in, ch_out, kernel, stride, padding
[ -1, 'Conv', [32, 64, 3, 2]],    #1  1/4   ch_in, ch_out, kernel, stride, padding, groups
[ -1, 'BottleneckCSP', [64, 64, 1]],  #2   ch_in, ch_out, number, shortcut, groups, expansion
[ -1, 'Conv', [64, 128, 3, 2]],   #3  1/8
[ -1, 'BottleneckCSP', [128, 128, 3]],    #4
[ -1, 'Conv', [128, 256, 3, 2]],  #5   1/16
[ -1, 'BottleneckCSP', [256, 256, 3]],    #6
[ -1, 'Conv', [256, 512, 3, 2]],  #7   1/32  

[ -1, 'SPP', [512, 512, [5, 9, 13]]],     #8

# FPN
[ -1, 'BottleneckCSP', [512, 512, 1, False]],     #9
[ -1, 'Conv',[512, 256, 1, 1]],   #10

[ -1, 'Upsample', [None, 2, 'nearest']],  #11   1/16 
[ [-1, 6], 'Concat', [1]],    #12    
[ -1, 'BottleneckCSP', [512, 256, 1, False]], #13
[ -1, 'Conv', [256, 128, 1, 1]],  #14
[ -1, 'Upsample', [None, 2, 'nearest']],  #15   1/8
[ [-1,4], 'Concat', [1]],     #16   1/8    #Encoder  FPN

[ -1, 'BottleneckCSP', [256, 128, 1, False]],     #17
[ -1, 'Conv', [128, 128, 3, 2]],      #18   1/16
[ [-1, 14], 'Concat', [1]],       #19
[ -1, 'BottleneckCSP', [256, 256, 1, False]],     #20
[ -1, 'Conv', [256, 256, 3, 2]],      #21   1/32
[ [-1, 10], 'Concat', [1]],   #22
[ -1, 'BottleneckCSP', [512, 512, 1, False]],     #23

[ [17, 20, 23], 'Detect',  [cfg.NUM_CLASSES, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [128, 256, 512]]], #Detection head 24

[ 16, 'Conv', [256, 128, 3, 1]],   #25  1/8  连接FPN的输出
[ -1, 'Upsample', [None, 2, 'nearest']],  #26  1/4
[ -1, 'BottleneckCSP', [128, 64, 1, False]],  #27
[ -1, 'Conv', [64, 32, 3, 1]],    #28
[ -1, 'Upsample', [None, 2, 'nearest']],  #29  1/
[ -1, 'Conv', [32, 16, 3, 1]],    #30
[ -1, 'BottleneckCSP', [16, 8, 1, False]],    #31
[ -1, 'Upsample', [None, 2, 'nearest']],  #32  原图尺寸
[ -1, 'Conv', [8, 2, 3, 1]], #33 Driving area segmentation head

[ 16, 'Conv', [256, 128, 3, 1]],   #34  1/8 连接FPN的输出
[ -1, 'Upsample', [None, 2, 'nearest']],  #35  1/4
[ -1, 'BottleneckCSP', [128, 64, 1, False]],  #36
[ -1, 'Conv', [64, 32, 3, 1]],    #37
[ -1, 'Upsample', [None, 2, 'nearest']],  #38  1/2
[ -1, 'Conv', [32, 16, 3, 1]],    #39
[ -1, 'BottleneckCSP', [16, 8, 1, False]],    #40
[ -1, 'Upsample', [None, 2, 'nearest']],  #41  原图尺寸
[ -1, 'Conv', [8, 2, 3, 1]] #42 Lane line segmentation head
]


YOLOPv2 = [
[37, 50, 62],                        #   Det_out_idx, Da_Segout_idx, LL_Segout_idx
[-1, 'Conv', [3, 32, 3, 1]],         #0  ch_in, ch_out, kernel, stride, padding, groups, act
[-1, 'Conv', [32, 64, 3, 2]],        #1  1/2
[-1, 'Conv', [64, 64, 3, 1]],        #2  
[-1, 'Conv', [64, 128, 3, 2]],       #3  1/4
[-1, 'ELAN', [128, 256, 2, False]],  #4  ch_in, ch_out, number, shortcut, groups, expansion
[-1, 'M2C', [256, 256, 0.5]],        #5  1/8 ch_in, ch_out, e
[-1, 'ELAN', [256, 512, 2, False]],  #6
[-1, 'M2C', [512, 512, 0.5]],        #7  1/16
[-1, 'ELAN', [512, 1024, 2, False]], #8
[-1, 'M2C', [1024, 1024]],           #9  1/32

[-1, 'ELAN', [1024, 1024, 2, False]],#10
[-1, 'Conv', [1024, 512, 1, 1]],     #11
[-1, 'Conv', [512, 512, 3, 1]],      #12
[-1, 'SPPF', [512, 512]],            #13  ch_in, ch_out, kernel
[-1, 'Conv', [512, 512, 3, 1]],      #14
[-5, 'Conv', [1024, 512, 1, 1]],     #15
[[-2, -1], 'Concat', [1]],           #16
[-1, 'Conv', [1024, 512, 1, 1]],     #17
[-1, 'Conv', [512, 256, 1, 1]],      #18
[-1, 'Upsample', [None, 2, 'nearest']], #19  1/16
[8, 'Conv', [1024, 256, 1, 1]],      #20
[[-2, -1], 'Concat', [1]],           #21  接seg分支
[-1, 'ELAN_X', [512, 256, 4]],       #22  ch_in, ch_out, number, expansion  
[-1, 'Conv', [256, 128, 1, 1]],      #23
[-1, 'Upsample', [None, 2, 'nearest']], #24  1/8
[6, 'Conv', [512, 128, 1, 1]],       #25
[[-2, -1], 'Concat', [1]],           #26 接lane line 分支
[-1, 'ELAN_X', [256, 128, 4]],       #27
[-1, 'Conv', [128, 256, 3, 1]],      #28  8倍检测
[-2, 'M2C', [128, 256, 0.5]],        #29  1/16
[[-8, -1], 'Concat', [1]],           #30
[-1, 'ELAN_X', [512, 256, 4]],       #31
[-1, 'Conv', [256, 512, 3, 1]],      #32  16倍检测
[-2, 'M2C', [256, 512, 0.5]],        #33  1/32
[[17, -1], 'Concat', [1]],           #34
[-1, 'ELAN_X', [1024, 512, 4]],      #35
[-1, 'Conv', [512, 1024, 3, 1]],     #36  32倍检测

# detect head
[[28, 32, 36], 'Detect',  [cfg.NUM_CLASSES, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [256, 512, 1024]]], #37

# driving area branch
[21, 'Conv', [512, 256, 3, 1]],      #38  1/16
[-1, 'Upsample', [None, 2, 'nearest']], #39  1/8
[-1, 'ResnetBolck', [256]],          #40
[-1, 'Conv', [128, 128, 1, 1]],      #41
[-1, 'Conv', [128, 64, 1, 1]],       #42
[-1, 'Upsample', [None, 2, 'nearest']], #43  1/4
[-1, 'Conv', [64, 32, 3, 1]],        #44
[-1, 'Upsample', [None, 2, 'nearest']], #45  1/2
[-1, 'Conv', [32, 16, 3, 1]],        #46
[-1, 'ResnetBolck', [16]],           #47
[-1, 'Conv', [8, 8, 1, 1]],          #48
[-1, 'Upsample', [None, 2, 'nearest']], #49  原图尺寸
[-1, 'Conv', [8, 2, 3, 1, None, 1, False]], #50   接sigmoid

# lane branch
[26, 'CBAM', [256, 16, 7]],          #51  1/8 ch_in, ratio, kernel_size
# [26, 'Conv', [256, 256, 3, 1]],      #51
[-1, 'Conv', [256, 128, 3, 1]],      #52
[-1, 'ConvTranspose', [128, 128, 2, 2]], #53  1/4 ch_in, ch_out, kernel, stride, act
[-1, 'ResnetBolck', [128]],          #54
[-1, 'Conv', [64, 64, 1, 1]],        #55
[-1, 'Conv', [64, 32, 3, 1]],        #56
[-1, 'ConvTranspose', [32, 32, 2, 2]],  #57  1/2
[-1, 'Conv', [32, 16, 3, 1]],        #58
[-1, 'ResnetBolck', [16]],           #59
[-1, 'Conv', [8, 8, 1, 1]],          #60
[-1, 'ConvTranspose', [8, 8, 2, 2, False]], #61  原图尺寸
[-1, 'Conv', [8, 2, 3, 1, None, 1, False]]  #62  接sigmoid
]

# tr = 1.0
# tr = 0.5
tr = cfg.MODEL.RATIO
# tr = 0.125
# mr = 0.75

YOLOPv2_tiny = [
[37, 50, 62],                              #   Det_out_idx, Da_Segout_idx, LL_Segout_idx
[-1, 'Conv', [3, int(32*tr), 3, 1]],            #0  ch_in, ch_out, kernel, stride, padding, groups, act
[-1, 'Conv', [int(32*tr), int(64*tr), 3, 2]],        #1  1/2
[-1, 'Conv', [int(64*tr), int(64*tr), 3, 1]],        #2  
[-1, 'Conv', [int(64*tr), int(128*tr), 3, 2]],       #3  1/4
[-1, 'ELAN', [int(128*tr), int(256*tr), 2, False]],  #4  ch_in, ch_out, number, shortcut, groups, expansion
[-1, 'M2C', [int(256*tr), int(256*tr), 0.5]],        #5  1/8 ch_in, ch_out, e
[-1, 'ELAN', [int(256*tr), int(512*tr), 2, False]],  #6
[-1, 'M2C', [int(512*tr), int(512*tr), 0.5]],        #7  1/16
[-1, 'ELAN', [int(512*tr), int(1024*tr), 2, False]], #8
[-1, 'M2C', [int(1024*tr), int(1024*tr)]],           #9  1/32

[-1, 'ELAN', [int(1024*tr), int(1024*tr), 2, False]],#10
[-1, 'Conv', [int(1024*tr), int(512*tr), 1, 1]],     #11
[-1, 'Conv', [int(512*tr), int(512*tr), 3, 1]],      #12
[-1, 'SPPF', [int(512*tr), int(512*tr)]],            #13  ch_in, ch_out, kernel
[-1, 'Conv', [int(512*tr), int(512*tr), 3, 1]],      #14
[-5, 'Conv', [int(1024*tr), int(512*tr), 1, 1]],     #15
[[-2, -1], 'Concat', [1]],                 #16
[-1, 'Conv', [int(1024*tr), int(512*tr), 1, 1]],     #17
[-1, 'Conv', [int(512*tr), int(256*tr), 1, 1]],      #18
[-1, 'Upsample', [None, 2, 'nearest']],    #19  1/16
[8, 'Conv', [int(1024*tr), int(256*tr), 1, 1]],      #20
[[-2, -1], 'Concat', [1]],                 #21  接seg分支
[-1, 'ELAN_X', [int(512*tr), int(256*tr), 4]],       #22  ch_in, ch_out, number, expansion  
[-1, 'Conv', [int(256*tr), int(128*tr), 1, 1]],      #23
[-1, 'Upsample', [None, 2, 'nearest']],    #24  1/8
[6, 'Conv', [int(512*tr), int(128*tr), 1, 1]],       #25
[[-2, -1], 'Concat', [1]],                 #26 接lane line 分支
[-1, 'ELAN_X', [int(256*tr), int(128*tr), 4]],       #27
[-1, 'Conv', [int(128*tr), int(256*tr), 3, 1]],      #28  8倍检测
[-2, 'M2C', [int(128*tr), int(256*tr), 0.5]],        #29  1/16
[[-8, -1], 'Concat', [1]],                 #30
[-1, 'ELAN_X', [int(512*tr), int(256*tr), 4]],       #31
[-1, 'Conv', [int(256*tr), int(512*tr), 3, 1]],      #32  16倍检测
[-2, 'M2C', [int(256*tr), int(512*tr), 0.5]],        #33  1/32
[[17, -1], 'Concat', [1]],                 #34
[-1, 'ELAN_X', [int(1024*tr), int(512*tr), 4]],      #35
[-1, 'Conv', [int(512*tr), int(1024*tr), 3, 1]],     #36  32倍检测

# detect head
[[28, 32, 36], 'Detect',  [cfg.NUM_CLASSES, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [int(256*tr), int(512*tr), int(1024*tr)]]], #37

# driving area branch
[21, 'Conv', [int(512*tr), int(256*tr), 3, 1]],      #38  1/16
[-1, 'Upsample', [None, 2, 'nearest']],              #39  1/8
[-1, 'ResnetBolck', [int(256*tr)]],                  #40
[-1, 'Conv', [int(128*tr), int(128*tr), 1, 1]],      #41
[-1, 'Conv', [int(128*tr), int(64*tr), 1, 1]],       #42
[-1, 'Upsample', [None, 2, 'nearest']],    #43  1/4
[-1, 'Conv', [int(64*tr), 32, 3, 1]],           #44
[-1, 'Upsample', [None, 2, 'nearest']],    #45  1/2
[-1, 'Conv', [32, 16, 3, 1]],              #46
[-1, 'ResnetBolck', [16]],                 #47
[-1, 'Conv', [8, 8, 1, 1]],                #48
[-1, 'Upsample', [None, 2, 'nearest']],    #49  原图尺寸
[-1, 'Conv', [8, 2, 3, 1, None, 1, False]],#50   接sigmoid

# lane branch
[26, 'CBAM', [int(256*tr), 16, 7]],                      #51  1/8 ch_in, ratio, kernel_size
# [26, 'Conv', [256, 256, 3, 1]],      #51
[-1, 'Conv', [int(256*tr), int(128*tr), 3, 1]],          #52
[-1, 'ConvTranspose', [int(128*tr), int(128*tr), 2, 2]], #53  1/4 ch_in, ch_out, kernel, stride, act
[-1, 'ResnetBolck', [int(128*tr)]],                      #54
[-1, 'Conv', [int(64*tr), int(64*tr), 1, 1]],            #55
[-1, 'Conv', [int(64*tr), int(32*tr), 3, 1]],            #56
[-1, 'ConvTranspose', [int(32*tr), int(32*tr), 2, 2]],   #57  1/2
[-1, 'Conv', [int(32*tr), 16, 3, 1]],                    #58
[-1, 'ResnetBolck', [16]],                               #59
[-1, 'Conv', [8, 8, 1, 1]],                              #60
[-1, 'ConvTranspose', [8, 8, 2, 2, False]],              #61  原图尺寸
[-1, 'Conv', [8, 2, 3, 1, None, 1, False]]               #62  接sigmoid
]


YOLOv8 = [
[23, 35],                                         #Det_out_idx, Da_Segout_idx, LL_Segout_idx
# Encoder
# from, module, args
[-1, 'Conv', [3, 8, 3, 2]],                        #0 1/2 ch_in, ch_out, kernel, stride, padding, groups
[-1, 'Conv', [8, int(32*tr), 3, 1]],               #1 
[-1, 'Conv', [int(32*tr), int(64*tr), 3, 2]],      #2 1/4
[-1, 'C2f', [int(64*tr), int(64*tr), 3, True]],    #3     ch_in, ch_out, number, shortcut, groups, expansion   
[-1, 'Conv', [int(64*tr), int(128*tr), 3, 2]],     #4 1/8   
[-1, 'C2f', [int(128*tr), int(128*tr), 6, True]],  #5 
[-1, 'Conv', [int(128*tr), int(256*tr), 3, 2]],    #6 1/16
[-1, 'C2f', [int(256*tr), int(256*tr), 6, True]],  #7
[-1, 'Conv', [int(256*tr), int(512*tr), 3, 2]],    #8 1/32
[-1, 'C2f', [int(512*tr), int(256*tr), 6, True]],  #9
[-1, 'SPPF', [int(256*tr), int(256*tr), 5]],       #10   ch_in, ch_out, kernel  

# FPN
[-1, 'Upsample', [None, 2, 'nearest']],            #11 1/16
[[-1, 7], 'Concat', [1]],                          #12
[-1, 'C2f', [int(512*tr), int(256*tr), 3, False]], #13

[-1, 'Upsample', [None, 2, 'nearest']],            #14 1/8
[[-1, 5], 'Concat', [1]],                          #15
[-1, 'C2f', [int(384*tr), int(128*tr), 3, False]], #16

[-1, 'Conv', [int(128*tr), int(256*tr), 3, 2]],    #17 1/16
[[-1, 13], 'Concat', [1]],                         #18
[-1, 'C2f', [int(512*tr), int(256*tr), 3, False]], #19

[-1, 'Conv', [int(256*tr), int(512*tr), 3, 2]],    #20 1/32
[[-1, 10], 'Concat', [1]],                         #21
[-1, 'C2f', [int(768*tr), int(512*tr), 3, False]], #22

# detect head
[[16, 19, 22], 'Detect',  [cfg.NUM_CLASSES, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [int(128*tr), int(256*tr), int(512*tr)]]], #23

# driving area segmentation branch
[6, 'Conv', [int(256*tr), int(128*tr), 3, 1]],          #24 1/16
[-1, 'Upsample', [None, 2, 'nearest']],                 #25 1/8
[[-1, 5], 'Concat', [1]],                               #26
[-1, 'C2f', [int(256*tr), int(64*tr), 3, True]],        #27

[-1, 'Upsample', [None, 2, 'nearest']], #28 1/4
[[-1, 3], 'Concat', [1]],               #29
[-1, 'C2f', [int(128*tr), 32, 3, True]],        #30

[-1, 'Upsample', [None, 2, 'nearest']], #31 1/2
[[-1, 1], 'Concat', [1]],               #32
[-1, 'C2f', [32+int(32*tr), 16, 1, True]],         #33

[ -1, 'Upsample', [None, 2, 'nearest']],#34  原图尺寸
[ -1, 'Conv', [16, 2, 3, 1]],           #35 

]


YOLOv8n = [
[22, 34], # Det_out_idx, Da_Segout_idx, LL_Segout_idx
# Encoder
# from, module, args
[-1, 'Conv', [3, 16, 3, 2]],                        # 0 1/2 ch_in, ch_out, kernel, stride, padding, groups
[-1, 'Conv', [16, 32, 3, 2]],                       # 1 1/4
[-1, 'C2f', [32, 32, 1, True]],                     # 2 ch_in, ch_out, number, shortcut, groups, expansion   
[-1, 'Conv', [32, 64, 3, 2]],                       # 3 1/8
[-1, 'C2f', [64, 64, 2, True]],                     # 4 ch_in, ch_out, number, shortcut, groups, expansion   
[-1, 'Conv', [64, 128, 3, 2]],                      # 5 1/16
[-1, 'C2f', [128, 128, 2, True]],                   # 6 ch_in, ch_out, number, shortcut, groups, expansion   
[-1, 'Conv', [128, 256, 3, 2]],                     # 7 1/32
[-1, 'C2f', [256, 256, 1, True]],                   # 8 ch_in, ch_out, number, shortcut, groups, expansion   
[-1, 'SPPF', [256, 256, 5]],                        # 9 ch_in, ch_out, kernel  

# FPN
[-1, 'Upsample', [None, 2, 'nearest']],             # 10 1/16
[[-1, 6], 'Concat', [1]],                           # 11
[-1, 'C2f', [384, 128, 1, False]],                  # 12

[-1, 'Upsample', [None, 2, 'nearest']],             # 13 1/8
[[-1, 4], 'Concat', [1]],                           # 14
[-1, 'C2f', [192, 64, 1, False]],                   # 15

[-1, 'Conv', [64, 64, 3, 2]],                       # 16 1/16
[[-1, 12], 'Concat', [1]],                          # 17
[-1, 'C2f', [192, 128, 1, False]],                  # 18

[-1, 'Conv', [128, 128, 3, 2]],                     # 19 1/32
[[-1, 9], 'Concat', [1]],                           # 20
[-1, 'C2f', [384, 256, 1, False]],                  # 21

# detect
[[15, 18, 21], 'Detect',  [cfg.NUM_CLASSES, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [64, 128, 256]]], # 22

# driving area segmentation branch
[18, 'Conv', [128, 128, 3, 1]],                         # 23 1/16
[-1, 'Upsample', [None, 2, 'nearest']],                 # 24 1/8
[[-1, 15, 4], 'Concat', [1]],                           # 25
[-1, 'C2f', [256, 128, 1, False]],                      # 26

[-1, 'Upsample', [None, 2, 'nearest']],                 # 27 1/4
[[-1, 2], 'Concat', [1]],                               # 28
[-1, 'C2f', [160, 64, 1, False]],                       # 29

[-1, 'Upsample', [None, 2, 'nearest']],                 # 30 1/2
[[-1, 0], 'Concat', [1]],                               # 31
[-1, 'C2f', [80, 16, 1, False]],                        # 32

[ -1, 'Upsample', [None, 2, 'nearest']],                # 33  原图尺寸
[ -1, 'Conv', [16, 2, 3, 1]],                           # 34 

]
