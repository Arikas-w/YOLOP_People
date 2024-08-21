# coding:UTF-8 

import cv2
import math
import numpy as np
# from pycubicspline import calc_2d_spline_interpolation
import os
import time
import torch
import torchvision


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, labels=()):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    nc = prediction.shape[1] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6))]
    # Apply constraints
    # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height

    x = prediction[xc]  # confidence

    # If none remain process next image
    if not x.shape[0]:
        return output

    # Compute conf
    x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

    # Box (center x, center y, width, height) to (x1, y1, x2, y2)
    box = xywh2xyxy(x[:, :4])

    # Detections matrix nx6 (xyxy, conf, cls)
    if multi_label:
        i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
        # i, j = (x[:, 5:] > conf_thres).nonzero().T
        x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
    else:  # best class only
        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

    # Filter by class
    if classes is not None:
        x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

    # Apply finite constraint
    # if not torch.isfinite(x).all():
    #     x = x[torch.isfinite(x).all(1)]

    # Check shape
    n = x.shape[0]  # number of boxes
    if not n:  # no boxes
        return output
    elif n > max_nms:  # excess boxes
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

    # Batched NMS
    c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
    boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
    i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
    if i.shape[0] > max_det:  # limit detections
        i = i[:max_det]
    
    output = x[i]
    if (time.time() - t) > time_limit:
        print(f'WARNING: NMS time limit {time_limit}s exceeded')
        return output  # time limit exceeded

    return output

def getFilePaths(rootPath):
    imgPaths = []
    for root, dirs, files in os.walk(rootPath):
        for f in files:
            imgPaths.append(os.path.join(root, f))
        
        for d in dirs:
            # subImgPaths = []
            subImgPaths = getFilePaths(os.path.join(root, d))
            if subImgPaths is not None:
                imgPaths += subImgPaths
                # subImgPaths = []

        return imgPaths

class curve():
    def __init__(self) -> None:
        self.exist = False
        self.p1 = None
        self.p2 = None
        self.p3 = None
        self.p4 = None
        self.p5 = None
        self.intersect = 0.0
        self.angle = 0.0
        self.length = 0.0
        self.curveType = -1  # 0 直线， 1 曲线
        self.birth = 0
        self.die = 0

class approxDP():
    def __init__(self) -> None:
        self.points = []
        self.length = []


def getMaxMinCoor(contour):
    xmin = contour[0][0]
    xmax = contour[0][0]
    ymin = contour[0][1]
    ymax = contour[0][1]
    pt_top = contour[0]
    pt_bottom = contour[0]
    pt_left = contour[0]
    pt_right = contour[0]
    for i in range(len(contour)):
        if contour[i][0] < xmin:
            xmin = contour[i][0]
            pt_left = contour[i]
        if contour[i][0] > xmax:
            xmax = contour[i][0]
            pt_right = contour[i]
        if contour[i][1] < ymin:
            ymin = contour[i][1]
            pt_top = contour[i]
        if contour[i][1] > ymax:
            ymax = contour[i][1]
            pt_bottom = contour[i]
    return xmin, xmax, ymin, ymax, pt_bottom, pt_top, pt_left, pt_right


def Polynomial_curve_fit(pts, n):
    
    # 构造矩阵X
    X = np.zeros((n + 1, n + 1), dtype=np.float64)
    for i in range(n + 1):
        for j in range(n + 1):
            for k in range(len(pts)):
                X[i, j] = X[i, j] + math.pow(pts[k][0], i + j)
    
    # 构造矩阵Y
    Y = np.zeros((n + 1, 1), dtype=np.float64)
    for i in range(n + 1):
        for k in range(len(pts)):
            Y[i, 0] = Y[i, 0] + math.pow(pts[k][0], i) * pts[k][1]

    ret, A = cv2.solve(X, Y)
    return A



def fitCubic(img_draw, ll_pts, width, height):
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
    for pts in ll_pts:
        approx = cv2.approxPolyDP(pts, 10, False)
        if approx.shape[0] < 2:
            continue

        # print(f"approx: f{approx.shape}")
        # 过滤斜率较小的直线
        if approx.shape[0] == 2:
            k, b = cal_line_coef(approx[0][0].tolist(), approx[1][0].tolist())
            angle = np.arctan(k) * 180 / np.pi
            if abs(angle) < 15:
                continue

            if(math.sqrt(math.pow(approx[0][0][0] - approx[1][0][0], 2) + math.pow(approx[0][0][1] - approx[1][0][1], 2)) < 50.0):
                continue
        
        pts = pts.reshape(pts.shape[0], 2).tolist()
        curve = dict()
        res = Polynomial_curve_fit(pts, 3)
        xmin, xmax, ymin, ymax, pt_bottom, pt_top, pt_top, pt_left, pt_right = getMaxMinCoor(pts)
        curve['xmin'] = xmin
        curve['xmax'] = xmax
        curve['d'] = res[0, 0]
        curve['c'] = res[1, 0]
        curve['b'] = res[2, 0]
        curve['a'] = res[3, 0]

        curve = NormalizeCurveCoef(curve, width, height)
        
        pts_num = 20
        interval = (curve['xmax'] - curve['xmin']) / pts_num
        # color = colors[i % len(colors)]
        points_fitted = []
        for j in range(pts_num):
            x = curve['xmin'] + j * interval
            y = curve['a'] * math.pow(x, 3) + curve['b'] * math.pow(x, 2) + curve['c'] * x + curve['d']

            x *= img_draw.shape[1]
            y *= img_draw.shape[0]

            points_fitted.append(tuple([int(x), int(y)]))
            cv2.circle(img_draw, (int(x), int(y)), 6, (0, 0, 255), -1)

        for n in range(1, len(points_fitted)):
            cv2.line(img_draw, points_fitted[n-1], points_fitted[n], (0, 255, 255), 2, 8, 0)

    return img_draw
    

def NormalizeCurveCoef(curve, width, height):
    curve['a'] = curve['a'] * math.pow(width, 3) / height
    curve['b'] = curve['b'] * math.pow(width, 2) / height
    curve['c'] = curve['c'] * width / height
    curve['d'] = curve['d'] / height
    curve['xmin'] = curve['xmin'] / width
    curve['xmax'] = curve['xmax'] / width
    
    return curve

def findNearPt(mask, pt, distance=10):
    find_flag = False
    middle = []

    right = False
    left = False
    position = False
    for i in range(1, 5):
        if (pt[0] + i) >= mask.shape[1]:  # 图像右边
            right = True
            position = True
            break
        elif (pt[0] - i) <= 0:
            left = True
            position = True
            break
        elif mask[pt[1], pt[0]+i] == 0: # 点在右侧， 需要向左搜索
            right = True
            position = True
            break
        elif mask[pt[1], pt[0]-i] == 0: 
            left = True
            position = True
            break

    # if not position:
    if False:
        # img = cv2.imread("./inference/output/ll_output_net.png")
        # cv2.circle(img, tuple(pt), 2, (0, 0, 255), -1)
        # cv2.imwrite("./inference/output/ll_test.png", img)
        # print("error point.")

        for i in range(distance):
            if (pt[0]+i) >= mask.shape[1]:
                if not find_flag :
                    middle = [pt[0] + i / 2, pt[1]]
                    find_flag = True
                return True, [int(m) for m in middle]
            if (mask[pt[1], pt[0]+i] == 0):
                find_flag = True
                middle = [pt[0] + i / 2, pt[1]]
                return True, [int(m) for m in middle]
        
        for i in range(distance):
            if (pt[0] - i) <= 0:
                if not find_flag :
                    middle = [pt[0] - i / 2, pt[1]]
                    find_flag = True
                    return True, [int(m) for m in middle]
            if (mask[pt[1], pt[0]-i] == 0):
                find_flag = True
                middle = [pt[0] - i / 2, pt[1]]
                return True, [int(m) for m in middle]
        
        return False, pt


    if left:
        for i in range(distance):
            if (pt[0]+i) >= mask.shape[1]:
                if not find_flag :
                    middle = [pt[0] + i / 2, pt[1]]
                    find_flag = True
                break
            if (mask[pt[1], pt[0]+i] == 0):
                find_flag = True
                middle = [pt[0] + i / 2, pt[1]]
                break
    
    if right:
        for i in range(distance):
            if (pt[0] - i) <= 0:
                if not find_flag :
                    middle = [pt[0] - i / 2, pt[1]]
                    find_flag = True
                break
            if (mask[pt[1], pt[0]-i] == 0):
                find_flag = True
                middle = [pt[0] - i / 2, pt[1]]
                break
    
    return find_flag, [int(m) for m in middle]




def FindPts(ll_mask):
    
    # ll_mask_net = cv2.imread("./inference/output/ll_output_net.png")
    # ll_mask = cv2.medianBlur(ll_mask, 5)
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    ll_open = cv2.morphologyEx(ll_mask, cv2.MORPH_OPEN, element)
    # element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # ll_open = cv2.morphologyEx(ll_open, cv2.MORPH_ERODE, element)
    ll_open = cv2.medianBlur(ll_open, 7)
    # ll_gradient = cv2.morphologyEx(ll_open, cv2.MORPH_GRADIENT, element)
    cv2.imwrite("./inference/output/ll_open.png", ll_open)
    # cv2.imwrite("./inference/output/ll_gradient.png", ll_gradient)

    img_draw = cv2.imread("./inference/output/ll_open.png")
    img_draw_copy = img_draw.copy()

    contours, hierarchy = cv2.findContours(ll_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    ll_pts = []
    for i in range(len(contours)):  
        color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        cv2.putText(img_draw, f"{i}", contours[i][0][0], cv2.FONT_HERSHEY_PLAIN, 1.0, color)

        # if i == 4:
        #     print("xxxx")
        ll_pt = []

        count = 0
        for j in range(int(contours[i].shape[0])):
            p1 = contours[i][j][0].tolist()
            # if i == 6 and p1[1] > 300:
            #     print("step")
            # p2 = contours[i][contours[i].shape[0] -1 - j][0]
            # p = [int((p1[n]+p2[n])/2) for n in range(2)]
            cv2.circle(img_draw_copy, tuple(p1), 1, (0, 0, 255), -1)
            # cv2.putText(img_draw_copy, str(j), tuple(p1), cv2.FONT_HERSHEY_TRIPLEX, 0.3, (0,0,255), 1)

            ret, p = findNearPt(ll_open, p1, 25)
            if ret:
                count = 0
                # if p[0] in x_labels:
                #     continue
                # x_labels.append(p[0])
                ll_pt.append(np.array(p))
                cv2.circle(img_draw, tuple(p), 1, (0, 0, 255), -1)
            else:
                count += 1
                cv2.circle(img_draw_copy, tuple(p1), 1, (0, 255, 255), -1)
            
            if count >= 2:
                if len(ll_pt) > 5:
                    ll_pts.append(np.array(ll_pt).reshape(len(ll_pt), 1, 2))
                    ll_pt = []
                count = 0
            
        if len(ll_pt) > 5: 
            ll_pts.append(np.array(ll_pt).reshape(len(ll_pt), 1, 2))
    cv2.imwrite("./inference/output/ll_contours.jpg", img_draw)
    cv2.imwrite("./inference/output/ll_contours_source.jpg", img_draw_copy)
    
    # return ll_pts
    ll_mask = cv2.imread("./inference/output/ll_output_net.png")
    for i, pts in enumerate(ll_pts):
        ll = ll_mask.copy()
        approx = cv2.approxPolyDP(pts, 5, True)
        if approx.shape[0] < 2:
            continue
        
        approx = nms_pt(approx, 5)
        for n in range(approx.shape[0]):
            p = approx[n][0].tolist()
            cv2.circle(ll, tuple(p), 3, (0, 0, 255), -1)
            cv2.putText(ll, f"{n}", tuple([p[0]+5, p[1]]), cv2.FONT_HERSHEY_TRIPLEX, 0.3, (0, 255, 255), 1)
        # cv2.imwrite(f"./inference/output/ll_mask_draw_{i}.png", ll)

    # cv2.imwrite(f"./inference/output/ll_mask_draw_{i}.png", ll)

    return ll_pts


def quick_sort(lists, i, j):
    if i >= j:
        return lists
    pivot = lists[i]
    low = i
    high = j
    while i < j:
        while i < j and lists[j][1] <= pivot[1]:
            j -= 1
        lists[i]=lists[j]
        while i < j and lists[i][1] >=pivot[1]:
            i += 1
        lists[j]=lists[i]
    lists[j] = pivot
    quick_sort(lists, low, i-1)
    quick_sort(lists, i+1, high)
    return lists


def nms_pt(approx, thresh=10):
    approx_list = approx.reshape(approx.shape[0], 2).tolist()
    
    if True:
        flag = False
        while (len(approx_list) > 1 and not flag):
            find = False
            for i in range(0, len(approx_list)):
                if find:
                    break
                for j in range(i+1, len(approx_list)):
                    if cal_length(approx_list[i], approx_list[j]) <= thresh:
                        
                        pt = []
                        pt.append((approx_list[i][0] + approx_list[j][0]) / 2)
                        pt.append((approx_list[i][1] + approx_list[j][1]) / 2)
                        approx_list.pop(max(i, j))
                        approx_list.pop(min(i, j))
                        approx_list.append([int(m) for m in pt])
                        find = True
                        break
            
            if find:
                flag = False
            else:
                flag = True
    
    approx_list = quick_sort(approx_list, 0, len(approx_list) - 1)
    
    return np.array(approx_list).reshape(len(approx_list), 1, 2)





def FindPts_old(ll_mask, filter=False, min_lane_thresh=50.0):
    
    # ll_mask_net = cv2.imread("./inference/output/ll_output_net.png")

    # element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # ll_open = cv2.morphologyEx(ll_mask, cv2.MORPH_CLOSE, element)

    contours, hierarchy = cv2.findContours(ll_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    ll_pts = []
    for i in range(len(contours)):
        xmin, xmax, ymin, ymax, pt_bottom, pt_top, pt_left, pt_right = getMaxMinCoor(contours[i].reshape(contours[i].shape[0], 2).tolist())

        # filter short lines
        if filter:
            if(math.sqrt(math.pow(xmax - xmin, 2) + math.pow(ymax - ymin, 2)) < min_lane_thresh):
                continue
        
        # color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        # cv2.putText(ll_mask_net, f"{i}", contours[i][0][0], cv2.FONT_HERSHEY_PLAIN, 1.0, color)

        ll_pt = []
        for j in range(int(contours[i].shape[0] / 2)):
            p1 = contours[i][j][0]
            p2 = contours[i][contours[i].shape[0] -1 - j][0]
            p = [int((p1[n]+p2[n])/2) for n in range(2)]
            ll_pt.append(np.array(p))

            # cv2.circle(ll_mask_net, tuple(p1), 2, color, -1)
        if len(ll_pt) > 1: 
            ll_pts.append(np.array(ll_pt).reshape(len(ll_pt), 1, 2))
    # cv2.imwrite("./inference/output/ll_contours.jpg", ll_mask_net)

    return ll_pts

def cal_bezier3(t, p0, p1, p2, p3):
    return (1 - t)**3 * p0 + 3 * t * (1 - t)**2 * p1 + 3 * (1 - t) * t ** 2 * p2 + t**3 * p3


def cal_bezier2(t, p0, p1, p2):
    return (1 - t)**2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2


def cal_line_coef(p0, p1):
    k = (p0[1] - p1[1]) / ((p0[0] - p1[0]) * 1.0 + 0.00001)
    b = p0[1] - k * p0[0]
    return k, b

# 计算线段长度
def cal_length(p0, p1):
    return math.sqrt(math.pow(p0[0] - p1[0], 2) + math.pow(p0[1] - p1[1], 2))


# 计算法线
def calNormalLine(k, p):
    k_normal = -1.0 / (k + 0.00001)
    b = p[1] - k_normal * p[0]
    return k_normal, b


def findMiddlePt(pts, pt1, pt2):
    xmin, xmax = [pt1[0], pt2[0]] if pt1[0] <= pt2[0] else [pt2[0], pt1[0]]
    ymin, ymax = [pt1[1], pt2[1]] if pt1[1] <= pt2[1] else [pt2[1], pt1[1]]
    tmp_vec = []
    for pt in pts:
        if pt[0][0] >= xmin and pt[0][0] <= xmax and pt[0][1] >= ymin and pt[0][1] <= ymax:
            tmp_vec.append(pt[0])
    
    return tmp_vec[len(tmp_vec) // 2]


def getTransD(angle):
    if angle < 10:
        d = 0
    elif angle < 20:
        d = 5
    elif angle < 30:
        d = 10
    elif angle < 45:
        d = 15
    else:
        d = 20
    
    return d

def casePt2(approx):
    '''
    只有两个点的情况
    approx: list
    '''
    approx_list = approxDP()
    k, b = cal_line_coef(approx[0], approx[1])
    angle = np.arctan(k) * 180 / np.pi
    if abs(angle) < 15: # 角度过小
        return approx_list
    
    approx_list.points.append([approx[0], approx[1]])
    approx_list.length.append(cal_length(approx[0], approx[1]))
    return approx_list


def casePt3(approx):
    '''
    只有三个点的情况
    approx: list 
    '''
    approx_single = []
    xmin, xmax, ymin, ymax, pt_bottom, pt_top, pt_left, pt_right = getMaxMinCoor(approx)
    # pt_start = pt_left if pt_left[1] > pt_right[1] else pt_right
    pt_start = pt_bottom
    approx_single.append(pt_start)
    # 找到中间值
    pt_middle = []
    for i in range(3):
        if approx[i][1] > ymin and approx[i][1] < ymax:
            pt_middle = approx[i]
            approx_single.append(pt_middle)
            break
    # pt_end = pt_left if pt_left[1] < pt_right[1] else pt_right
    pt_end = pt_top
    approx_single.append(pt_end)
    # total_length = cal_length(pt_start, pt_middle) + cal_length(pt_middle, pt_end)

    return np.array(approx_single).reshape(len(approx_single), 1, 2)


# 两个二阶贝塞尔曲线拟合
def fitBezier2(img_draw, ll_pts, ratio=1.0):
    
    for pts in ll_pts:
        approx = cv2.approxPolyDP(pts, 10, False)
        if approx.shape[0] < 2:
            continue

        # print(f"approx: f{approx.shape}")
        # 过滤斜率较小的直线
        if approx.shape[0] == 2:
            k, b = cal_line_coef(approx[0][0].tolist(), approx[1][0].tolist())
            angle = np.arctan(k) * 180 / np.pi
            if abs(angle) < 15:
                continue

            if(math.sqrt(math.pow(approx[0][0][0] - approx[1][0][0], 2) + math.pow(approx[0][0][1] - approx[1][0][1], 2)) < 50.0):
                continue
        
        # 两个 二阶贝塞尔曲线拟合
        if approx.shape[0] == 2:
            ps0 = approx[0][0].tolist()
            ps1 = approx[1][0].tolist()
            k, b = cal_line_coef(ps0, ps1)
            step = (ps0[0] - ps1[0]) / 4.0
            p0 = ps0
            x_1_4 = ps0[0] - step
            p1 = [x_1_4, k * x_1_4 + b]
            x_2_4 = ps0[0] - step * 2
            p2 = [x_2_4, k * x_2_4 + b]
            x_3_4 = ps0[0] - step * 3
            p3 = [x_3_4, k * x_3_4 + b]
            p4 = ps1
        
        elif approx.shape[0] == 3:
            ps0 = approx[0][0].tolist()
            ps1 = approx[1][0].tolist()
            ps2 = approx[2][0].tolist()

            p0 = ps0
            p1 = findMiddlePt(pts, ps0, ps1).tolist()
            p2 = ps1
            p3 = findMiddlePt(pts, ps1, ps2).tolist()
            p4 = ps2
        
        else:
            continue
        
        T = np.linspace(0, 1, 200)
        for t in T:
            x1 = int(cal_bezier2(t, p0[0], p1[0], p2[0]) / ratio)
            y1 = int(cal_bezier2(t, p0[1], p1[1], p2[1]) / ratio)

            x2 = int(cal_bezier2(t, p2[0], p3[0], p4[0]) / ratio)
            y2 = int(cal_bezier2(t, p2[1], p3[1], p4[1]) / ratio)

            cv2.circle(img_draw, (x1, y1), 3, (0, 255, 255), -1)
            cv2.circle(img_draw, (x2, y2), 3, (0, 255, 255), -1)
        cv2.circle(img_draw, tuple([int(e / ratio) for e in p0]), 6, (0, 0, 255), -1)
        cv2.circle(img_draw, tuple([int(e / ratio) for e in p1]), 6, (0, 0, 255), -1)
        cv2.circle(img_draw, tuple([int(e / ratio) for e in p2]), 6, (0, 0, 255), -1)
        cv2.circle(img_draw, tuple([int(e / ratio) for e in p3]), 6, (0, 0, 255), -1)
        cv2.circle(img_draw, tuple([int(e / ratio) for e in p4]), 6, (0, 0, 255), -1)
        
    return img_draw 


# 三阶贝塞尔曲线拟合
def fitBezier3(img_draw, ll_pts, ratio=1.0, middle_x=320):
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
    length_thres = 77
    length_ratio =  1.2
    length_max_thresh = 180
    sim_angle = 6
    left_curve = curve()
    right_curve = curve()
    for n, pts in enumerate(ll_pts):
        approx = cv2.approxPolyDP(pts, 5, True)
        if approx.shape[0] < 2:
            continue

        approx = nms_pt(approx, 5)

        approx_list = approxDP()
        approx_single = []
        # if approx.shape[0] == 2:
        #     pass
        # elif approx.shape[0] == 3:
        #     approx_single, length = casePt3(approx.reshape(approx.shape[0], 2).tolist())
        #     approx_list.points.append(approx_single)
        #     approx_list.length.append(length)
        # elif approx.shape[0] == 4:
        #     pass

        process = False
        # if approx.shape[0] > 4:
        #     xmin, xmax, ymin, ymax, pt_bottom, pt_top, pt_left, pt_right = getMaxMinCoor(approx.reshape(approx.shape[0], 2).tolist())
        #     approx_single = []
        #     approx_single.append(pt_bottom)
        #     approx_single.append(pt_top)

        #     approx_list.points.append(approx_single)
        #     approx_list.length.append(cal_length(pt_bottom, pt_top))
        #     process = True
        
        if approx.shape[0] == 3: # L 型情况
            k01, b01 = cal_line_coef(approx[0][0].reshape(2).tolist(), approx[1][0].reshape(2).tolist())
            k02, b02 = cal_line_coef(approx[0][0].reshape(2).tolist(), approx[2][0].reshape(2).tolist())
            # 计算两条直线的夹角
            angle = np.arctan(abs((k01-k02)/(1+k01*k02))) * 180 / np.pi if k01*k02 != -1 else 90
            if abs(angle - 90) < 6 or abs(angle - 45) < 6:
                process = True
                # 选取长的线作为最终拟合直线
                length01 = cal_length(approx[0][0].reshape(2).tolist(), approx[1][0].reshape(2).tolist())
                length02 = cal_length(approx[0][0].reshape(2).tolist(), approx[2][0].reshape(2).tolist())
                approx_single = []
                if length01 > length02:
                    approx_single.append(approx[0][0].reshape(2).tolist())
                    approx_single.append(approx[1][0].reshape(2).tolist())
                    approx_list.points.append(approx_single)
                    approx_list.length.append(length01)
                else:
                    approx_single.append(approx[0][0].reshape(2).tolist())
                    approx_single.append(approx[2][0].reshape(2).tolist())
                    approx_list.points.append(approx_single)
                    approx_list.length.append(length02)
                

        if approx.shape[0] == 4:  # 处理T字形情况
            k03, b03 = cal_line_coef(approx[0][0].reshape(2).tolist(), approx[3][0].reshape(2).tolist())
            angle03 = abs(np.arctan(k03) * 180 / np.pi)
            if angle03 < 15:
                T_flag = False
            else:
                T_flag = True
            if T_flag:
                k01, b01 = cal_line_coef(approx[0][0].reshape(2).tolist(), approx[1][0].reshape(2).tolist())
                k31, b31 = cal_line_coef(approx[3][0].reshape(2).tolist(), approx[1][0].reshape(2).tolist())
                angle01 = np.arctan(k01) * 180 / np.pi
                angle31 = np.arctan(k31) * 180 / np.pi
                if abs(angle01 - angle31) < 5:
                    approx_single = []
                    approx_single.append(approx[0][0].reshape(2).tolist())
                    approx_single.append(approx[1][0].reshape(2).tolist())
                    approx_single.append(approx[3][0].reshape(2).tolist())
                    approx_list.points.append(approx_single)
                    length1 = cal_length(approx[0][0].reshape(2).tolist(), approx[1][0].reshape(2).tolist())
                    length2 = cal_length(approx[3][0].reshape(2).tolist(), approx[1][0].reshape(2).tolist())
                    approx_list.length.append(length1+length2)
                    process = True
                
                if not process:
                    k02, b02 = cal_line_coef(approx[0][0].reshape(2).tolist(), approx[2][0].reshape(2).tolist())
                    k32, b32 = cal_line_coef(approx[3][0].reshape(2).tolist(), approx[2][0].reshape(2).tolist())
                    angle02 = np.arctan(k02) * 180 / np.pi
                    angle32 = np.arctan(k32) * 180 / np.pi
                    if abs(angle02 - angle32) < 5:
                        approx_single = []
                        approx_single.append(approx[0][0].reshape(2).tolist())
                        approx_single.append(approx[2][0].reshape(2).tolist())
                        approx_single.append(approx[3][0].reshape(2).tolist())
                        approx_list.points.append(approx_single)
                        length1 = cal_length(approx[0][0].reshape(2).tolist(), approx[2][0].reshape(2).tolist())
                        length2 = cal_length(approx[3][0].reshape(2).tolist(), approx[2][0].reshape(2).tolist())
                        approx_list.length.append(length1+length2)
                        process = True

        if not process:
        # else:    
            # 对多边形拟合后的点进行处理
            
            if approx.shape[0] == 3:
                approx = casePt3(approx.reshape(approx.shape[0], 2).tolist())
            

            total_length = 0.0 
            start = True
            for i in range(approx.shape[0]-1):
                # 过滤斜率较小的直线
                k, b = cal_line_coef(approx[i][0].tolist(), approx[i+1][0].tolist())
                angle = np.arctan(k) * 180 / np.pi
                if abs(angle) < 15:
                    if len(approx_single) > 0:
                        approx_list.points.append(approx_single)
                        approx_list.length.append(total_length)
                        approx_single = []
                        total_length = 0.0
                    start = True
                    continue
                if start:
                    approx_single.append(approx[i][0].tolist())
                    approx_single.append(approx[i+1][0].tolist())
                    total_length += cal_length(approx[i][0].tolist(), approx[i+1][0].tolist())
                    start = False
                else:
                    approx_single.append(approx[i+1][0].tolist())
                    total_length += cal_length(approx[i][0].tolist(), approx[i+1][0].tolist())

            if len(approx_single) > 0:
                approx_list.points.append(approx_single)
                approx_list.length.append(total_length)

        for i, approx_curve in enumerate(approx_list.points):        
            # 过滤短线
            if len(approx_curve) > 2:
                if(approx_list.length[i] < length_thres):
                    continue
            
            # 取最开始四个点
            if len(approx_curve) > 4:
                approx_curve = approx_curve[:4]

            
            # 两个点的情况
            if len(approx_curve) == 2:
                ps1 = approx_curve[0]
                ps2 = approx_curve[1]
                k, b = cal_line_coef(ps1, ps2)
                angle = np.arctan(k) * 180 / np.pi
                intersect = (360 - b) / (k + 0.00001)  # 与x轴的交点
                if k < 0 and intersect < (middle_x + 30): # left
                    # color = (0, 255, 255)
                    left_flag = True
                    right_flag = False
                elif k > 0 and intersect >= (middle_x - 30):  # right
                    # color = (255, 0, 255)
                    left_flag = False
                    right_flag = True
                else:
                    left_flag = False
                    right_flag = False            

                step = (ps1[0] - ps2[0]) / 4.0
                p1 = ps1
                x_1_4 = ps1[0] - step
                p2 = [x_1_4, k * x_1_4 + b]
                x_3_4 = ps1[0] - step * 3
                p3 = [x_3_4, k * x_3_4 + b]
                p4 = ps2

                # 是否需要添加判断左侧车道线不能在右侧  或者 右侧车道线不能在左侧？？？
                # 当车道线的长度很长，那么是否就可以100%确定是车道线而不是误检
                # 通过100%确定的车道线是否就可以删除其他不合理的车道线
                if left_flag:
                    if not left_curve.exist and approx_list.length[i] > length_thres:  # 第一条曲线
                        left_curve.exist = True
                        left_curve.p1 = p1
                        left_curve.p2 = p2 
                        left_curve.p3 = p3
                        left_curve.p4 = p4
                        left_curve.intersect = intersect
                        left_curve.angle = angle
                        left_curve.length = approx_list.length[i]
                        left_curve.curveType = 0
                    elif left_curve.exist:

                        # 判断两条直线的斜率是否一致
                        # 如果一致，则合并两条直线

                        # 角度相差不大，且是直线的情况
                        merge_flag = False
                        if p1[1] > left_curve.p1[1]: # Y值较大
                            k_new, b_new = cal_line_coef(p1, left_curve.p4)
                            angle_new = np.arctan(k_new) * 180 / np.pi
                            if abs(angle - left_curve.angle) < sim_angle and abs(angle_new - angle) < sim_angle and abs(angle_new - left_curve.angle) < sim_angle:
                                left_curve.p1 = p1
                                left_curve.length = cal_length(left_curve.p1, left_curve.p4)
                                left_curve.angle = angle_new
                                merge_flag = True
                        
                        if p4[1] < left_curve.p4[1] and left_curve.curveType == 0:
                            k_new, b_new = cal_line_coef(left_curve.p1, p4)
                            angle_new = np.arctan(k_new) * 180 / np.pi
                            if abs(angle - left_curve.angle) < sim_angle and abs(angle_new - angle) < sim_angle and abs(angle_new - left_curve.angle) < sim_angle:
                                left_curve.p4 = p4
                                left_curve.length = cal_length(left_curve.p1, left_curve.p4)
                                left_curve.angle = angle_new
                                merge_flag = True

                            
                        if not merge_flag and (((intersect > left_curve.intersect) and (approx_list.length[i] > 1.0 * left_curve.length)) \
                            or ((approx_list.length[i] > length_max_thresh) and intersect > left_curve.intersect)\
                                or (approx_list.length[i] > length_ratio * left_curve.length)):
                            # left_curve.exist = True
                            left_curve.p1 = p1
                            left_curve.p2 = p2
                            left_curve.p3 = p3
                            left_curve.p4 = p4
                            left_curve.intersect = intersect
                            left_curve.angle = angle
                            left_curve.length = approx_list.length[i]
                            left_curve.curveType = 0
                elif right_flag:
                    if not right_curve.exist and approx_list.length[i] > length_thres:  # 第一条曲线
                        right_curve.exist = True
                        right_curve.p1 = p1
                        right_curve.p2 = p2
                        right_curve.p3 = p3
                        right_curve.p4 = p4
                        right_curve.intersect = intersect
                        right_curve.angle = angle
                        right_curve.length = approx_list.length[i]
                        right_curve.curveType = 0
                    
                    elif right_curve.exist:

                        merge_flag = False
                        if p1[1] > right_curve.p1[1]: # Y值较大
                            k_new, b_new = cal_line_coef(p1, right_curve.p4)
                            angle_new = np.arctan(k_new) * 180 / np.pi
                            if abs(angle - right_curve.angle) < sim_angle and abs(angle_new - angle) < sim_angle and abs(angle_new - right_curve.angle) < sim_angle:
                                right_curve.p1 = p1
                                right_curve.length = cal_length(right_curve.p1, right_curve.p4)
                                right_curve.angle = angle_new
                                merge_flag = True
                        
                        if p4[1] < right_curve.p4[1] and right_curve.curveType == 0:
                            k_new, b_new = cal_line_coef(right_curve.p1, p4)
                            angle_new = np.arctan(k_new) * 180 / np.pi
                            if abs(angle - right_curve.angle) < sim_angle and abs(angle_new - angle) < sim_angle and abs(angle_new - right_curve.angle) < sim_angle:
                                right_curve.p4 = p4
                                right_curve.length = cal_length(right_curve.p1, right_curve.p4)
                                right_curve.angle = angle_new
                                merge_flag = True

                        # if abs(angle - right_curve.angle) < 2 and right_curve.curveType == 0:
                        #     if p1[1] > right_curve.p1[1]:
                        #         right_curve.p1 = p1
                        #         right_curve.length = cal_length(right_curve.p1, right_curve.p4)
                        #         k_new, b_new = cal_line_coef(right_curve.p1, right_curve.p4)
                        #         right_curve.angle = np.arctan(k_new) * 180 / np.pi
                            
                        #     if p4[1] < right_curve.p4[1]:
                        #         right_curve.p4 = p4
                        #         right_curve.length = cal_length(right_curve.p1, right_curve.p4)
                        #         k_new, b_new = cal_line_coef(right_curve.p1, right_curve.p4)
                        #         right_curve.angle = np.arctan(k_new) * 180 / np.pi

                        # 如果车道线更靠近车子
                        # if (intersect < right_curve.intersect) and (approx_list.length[i] > 1.0 * right_curve.length):  
                        if not merge_flag and ((intersect < right_curve.intersect and approx_list.length[i] > 1.0 * right_curve.length) \
                            or (approx_list.length[i] > length_max_thresh and intersect < right_curve.intersect) \
                                or (approx_list.length[i] > length_ratio * right_curve.length)):  
                            right_curve.p1 = p1
                            right_curve.p2 = p2
                            right_curve.p3 = p3
                            right_curve.p4 = p4
                            right_curve.intersect = intersect
                            right_curve.angle = angle
                            right_curve.length = approx_list.length[i]
                            right_curve.curveType = 0


            elif len(approx_curve) == 3: 
                ps1 = approx_curve[0]
                ps2 = approx_curve[1]
                ps3 = approx_curve[2]

                k1, b1 = cal_line_coef(ps1, ps2)
                step1 = (ps1[0] - ps2[0]) / 16.0
                p1 = ps1
                x_3_4 = ps1[0] - step1 * 14
                p2 = [x_3_4, k1 * x_3_4 + b1]

                k2, b2 = cal_line_coef(ps2, ps3)
                step2 = (ps2[0] - ps3[0]) / 16.0
                x_1_4 = ps2[0] - step2 * 4
                p3 = [x_1_4, k2 * x_1_4 + b2]

                # 两条直线之间的夹角
                angle = np.arctan(abs((k1-k2)/(1+k1*k2))) * 180 / np.pi if k1*k2 != -1 else 90
                # 计算p1 p2平行的直线，距离为d
                k3, b3 = cal_line_coef(p2, p3)
                d = getTransD(angle)

                # d = 10 if angle < 30 else 20
                c1 = d * math.sqrt(k3**2 + 1) - k3 * p3[0] + p3[1]
                c2 = -1 * d * math.sqrt(k3**2 + 1) - k3 * p3[0] + p3[1]
                
                if ps3[0] < ps2[0]: # 往左偏
                    c = max(c1, c2) if k3 < 0 else min(c1, c2)
                else: # 往右偏
                    c = min(c1, c2) if k3 < 0 else max(c1, c2)
                # c = c1 if c1 < c2 else c2
                # y = k3 * x + c
                # 计算经过点p2、 p3法线
                k_normal_1, b_normal_1 = calNormalLine(k3, p2)
                k_normal_2, b_normal_2 = calNormalLine(k3, p3)

                # 计算法线与平移线的 交点
                p2_x_new = (b_normal_1 - c) / (k3 - k_normal_1 + 0.00001)
                p2_y_new = k3 * p2_x_new + c 

                p3_x_new = (b_normal_2 - c) / (k3 - k_normal_2 + 0.00001)
                p3_y_new = k3 * p3_x_new + c 

                p2 = [p2_x_new, p2_y_new]
                p3 = [p3_x_new, p3_y_new]

                p4 = approx_curve[2]

                intersect = (360 - b1) / k1  # 与x轴的交点
                angle = np.arctan(k1) * 180 / np.pi
                if k1 < 0 and intersect < (middle_x + 30): # left
                    # color = (0, 255, 255)
                    left_flag = True
                    right_flag = False
                elif k1 > 0 and intersect >= (middle_x - 30):  # right
                    # color = (255, 0, 255)
                    left_flag = False
                    right_flag = True
                else:
                    left_flag = False
                    right_flag = False

                if left_flag:
                    if not left_curve.exist:  # 第一条曲线
                        left_curve.exist = True
                        left_curve.p1 = p1
                        left_curve.p2 = p2 
                        left_curve.p3 = p3
                        left_curve.p4 = p4
                        left_curve.intersect = intersect
                        left_curve.angle = angle
                        left_curve.length = approx_list.length[i]
                        left_curve.curveType = 0 if d == 0 else 1
                    else:
                        if (abs(intersect - left_curve.intersect) < 20 and (approx_list.length[i] > length_ratio * left_curve.length)) or \
                            (intersect > left_curve.intersect and (approx_list.length[i] > length_ratio * left_curve.length)):
                            # left_curve.exist = True 
                            left_curve.p1 = p1
                            left_curve.p2 = p2
                            left_curve.p3 = p3
                            left_curve.p4 = p4
                            left_curve.intersect = intersect
                            left_curve.angle = angle
                            left_curve.length = approx_list.length[i]
                            left_curve.curveType = 0 if d == 0 else 1
                elif right_flag:
                    if not right_curve.exist:  # 第一条曲线
                        right_curve.exist = True
                        right_curve.p1 = p1
                        right_curve.p2 = p2
                        right_curve.p3 = p3
                        right_curve.p4 = p4
                        right_curve.intersect = intersect
                        right_curve.angle = angle
                        right_curve.length = approx_list.length[i]
                        right_curve.curveType = 0 if d == 0 else 1
                    else:
                        # 如果车道线更靠近车子
                        if (abs(intersect - right_curve.intersect) < 20 and (approx_list.length[i] > length_ratio * right_curve.length)) or \
                            (intersect < right_curve.intersect and (approx_list.length[i] > length_ratio * right_curve.length)):  
                            # left_curve.exist = True
                            right_curve.p1 = p1
                            right_curve.p2 = p2
                            right_curve.p3 = p3
                            right_curve.p4 = p4
                            right_curve.intersect = intersect
                            right_curve.angle = angle
                            right_curve.length = approx_list.length[i]
                            right_curve.curveType = 0 if d == 0 else 1

            else:
                # 是否需要排序
                p1 = approx_curve[0]
                p2 = approx_curve[1]
                p3 = approx_curve[2]
                p4 = approx_curve[-1]

                # 计算靠近车子的直线斜率以及与x轴的交点
                k_p1p2, b_p1p2 = cal_line_coef(p1, p2)
                angle_slope = np.arctan(k_p1p2) * 180 / np.pi
                intersect = (360 - b_p1p2) / k_p1p2  # 与x轴的交点
                if k_p1p2 < 0 and intersect < (middle_x + 30): # left
                    # color = (0, 255, 255)
                    left_flag = True
                    right_flag = False
                elif k_p1p2 > 0 and intersect > (middle_x - 30):  # right
                    # color = (255, 0, 255)
                    left_flag = False
                    right_flag = True
                else:
                    left_flag = False
                    right_flag = False


                # k1, b1 = cal_line_coef(p2, p3)
                # middle_x = (p2[0] + p3[0]) / 2.0
                # middle_y =  k1 * middle_x + b1
                # p_middle = [middle_x, middle_y]

                # 计算两边端点的直线
                # k_p1p2, b_p1p2 = cal_line_coef(p1, p2)
                k_p4p3, b_p4p3 = cal_line_coef(p4, p3)

                # 两条直线之间的夹角
                angle = np.arctan(abs((k_p1p2-k_p4p3)/(1+k_p1p2*k_p4p3))) * 180 / np.pi if k_p1p2*k_p4p3 != -1 else 90

                # 计算p2 p3平行的直线，y = k3 * x + c, 距离为d
                k3, b3 = cal_line_coef(p2, p3)
                # d = 5 if angle < 30 else 20
                d = getTransD(angle)
                c1 = d * math.sqrt(k3**2 + 1) - k3 * p3[0] + p3[1]
                c2 = -1 * d * math.sqrt(k3**2 + 1) - k3 * p3[0] + p3[1]
                if p3[0] < p2[0]: # 往左偏
                    c = max(c1, c2) if k3 < 0 else min(c1, c2)
                else: # 往右偏
                    c = min(c1, c2) if k3 < 0 else max(c1, c2)

                # 计算经过点p2、 p3法线
                k_normal_1, b_normal_1 = calNormalLine(k3, p2)
                k_normal_2, b_normal_2 = calNormalLine(k3, p3)

                # 计算法线与平移线的 交点
                p2_x_new = (b_normal_1 - c) / (k3 - k_normal_1 + 0.00001)
                p2_y_new = k3 * p2_x_new + c 

                p3_x_new = (b_normal_2 - c) / (k3 - k_normal_2 + 0.00001)
                p3_y_new = k3 * p2_x_new + c 

                p2 = [p2_x_new, p2_y_new]
                p3 = [p3_x_new, p3_y_new]

                if left_flag:
                    if not left_curve.exist:  # 第一条曲线
                        left_curve.exist = True
                        left_curve.p1 = p1
                        left_curve.p2 = p2 
                        left_curve.p3 = p3
                        left_curve.p4 = p4
                        left_curve.intersect = intersect
                        left_curve.angle = angle_slope
                        left_curve.length = approx_list.length[i]
                        left_curve.curveType = 1
                    else:
                        if (intersect > left_curve.intersect) and (approx_list.length[i] > length_ratio * left_curve.length):
                            # left_curve.exist = True
                            left_curve.p1 = p1
                            left_curve.p2 = p2
                            left_curve.p3 = p3
                            left_curve.p4 = p4
                            left_curve.intersect = intersect
                            left_curve.angle = angle_slope
                            left_curve.length = approx_list.length[i]
                            left_curve.curveType = 1
                elif right_flag:
                    if not right_curve.exist:  # 第一条曲线
                        right_curve.exist = True
                        right_curve.p1 = p1
                        right_curve.p2 = p2
                        right_curve.p3 = p3
                        right_curve.p4 = p4
                        right_curve.intersect = intersect
                        right_curve.angle = angle_slope
                        right_curve.length = approx_list.length[i]
                        right_curve.curveType = 1
                    else:
                        # 如果车道线更靠近车子
                        if (intersect < right_curve.intersect) and (approx_list.length[i] > length_ratio * right_curve.length):  
                            # left_curve.exist = True
                            right_curve.p1 = p1
                            right_curve.p2 = p2
                            right_curve.p3 = p3
                            right_curve.p4 = p4
                            right_curve.intersect = intersect
                            right_curve.angle = angle_slope
                            right_curve.length = approx_list.length[i]
                            right_curve.curveType = 1

            # # draw point
            # T = np.linspace(0, 1, 200)
            # for t in T:
            #     x = int(cal_bezier3(t, p1[0], p2[0], p3[0], p4[0]) / ratio)
            #     y = int(cal_bezier3(t, p1[1], p2[1], p3[1], p4[1]) / ratio)

            #     cv2.circle(img_draw, (x, y), 3, (0, 255, 255), -1)
            
            # cv2.circle(img_draw, tuple([int(e / ratio) for e in p1]), 6, (0, 0, 255), -1)
            # cv2.circle(img_draw, tuple([int(e / ratio) for e in p2]), 6, (0, 0, 255), -1)
            # cv2.circle(img_draw, tuple([int(e / ratio) for e in p3]), 6, (0, 0, 255), -1)
            # cv2.circle(img_draw, tuple([int(e / ratio) for e in p4]), 6, (0, 0, 255), -1)
        
    if left_curve.exist:
        T = np.linspace(0, 1, 200)
        for t in T:
            x = int(cal_bezier3(t, left_curve.p1[0], left_curve.p2[0], left_curve.p3[0], left_curve.p4[0]) / ratio)
            y = int(cal_bezier3(t, left_curve.p1[1], left_curve.p2[1], left_curve.p3[1], left_curve.p4[1]) / ratio)

            cv2.circle(img_draw, (x, y), 3, (255, 0, 255), -1)
        
        cv2.circle(img_draw, tuple([int(e / ratio) for e in left_curve.p1]), 6, (0, 0, 255), -1)
        # cv2.circle(img_draw, tuple([int(e / ratio) for e in left_curve.p2]), 6, (0, 0, 255), -1)
        # cv2.circle(img_draw, tuple([int(e / ratio) for e in left_curve.p3]), 6, (0, 0, 255), -1)
        cv2.circle(img_draw, tuple([int(e / ratio) for e in left_curve.p4]), 6, (0, 0, 255), -1)

        cv2.putText(img_draw, f"left curve angle: {abs(left_curve.angle):.1f}", (20, 30), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 0, 255), 2)
    else:
        cv2.putText(img_draw, f"left curve angle: None", (20, 30), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 0, 255), 2)
    
    if right_curve.exist:
        T = np.linspace(0, 1, 200)
        for t in T:
            x = int(cal_bezier3(t, right_curve.p1[0], right_curve.p2[0], right_curve.p3[0], right_curve.p4[0]) / ratio)
            y = int(cal_bezier3(t, right_curve.p1[1], right_curve.p2[1], right_curve.p3[1], right_curve.p4[1]) / ratio)

            cv2.circle(img_draw, (x, y), 3, (0, 255, 255), -1)
        
        cv2.circle(img_draw, tuple([int(e / ratio) for e in right_curve.p1]), 6, (0, 0, 255), -1)
        # cv2.circle(img_draw, tuple([int(e / ratio) for e in right_curve.p2]), 6, (0, 0, 255), -1)
        # cv2.circle(img_draw, tuple([int(e / ratio) for e in right_curve.p3]), 6, (0, 0, 255), -1)
        cv2.circle(img_draw, tuple([int(e / ratio) for e in right_curve.p4]), 6, (0, 0, 255), -1)

        cv2.putText(img_draw, f"right curve angle: {abs(right_curve.angle):.1f}", (20, 60), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 0, 255), 2)
    else:
        cv2.putText(img_draw, f"right curve angle: None", (20, 60), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 0, 255), 2)
    return img_draw


def getLeftRightCurves(approx_list, length_thres=80, middle_x=320):
    left_curves = []
    right_curves = []
    for i, approx_curve in enumerate(approx_list.points):        
        if len(approx_curve) == 1:
            continue

        # 过滤短线
        if len(approx_curve) >= 2:
            if(approx_list.length[i] < length_thres):
                continue
        
        # 取最开始四个点
        if len(approx_curve) > 4:
            approx_curve = approx_curve[:4]

        
        # 两个点的情况
        if len(approx_curve) == 2:
            ps1 = approx_curve[0]
            ps2 = approx_curve[1]
            k, b = cal_line_coef(ps1, ps2)
            angle = np.arctan(k) * 180 / np.pi
            intersect = (360 - b) / (k + 0.00001)  # 与x轴的交点
            if k < 0 and intersect < (middle_x + 30): # left
                # color = (0, 255, 255)
                left_flag = True
                right_flag = False
            elif k > 0 and intersect >= (middle_x - 30):  # right
                # color = (255, 0, 255)
                left_flag = False
                right_flag = True
            else:
                left_flag = False
                right_flag = False            

            step = (ps1[0] - ps2[0]) / 4.0
            p1 = ps1
            x_1_4 = ps1[0] - step
            p2 = [x_1_4, k * x_1_4 + b]
            x_3_4 = ps1[0] - step * 3
            p3 = [x_3_4, k * x_3_4 + b]
            p4 = ps2

            # 是否需要添加判断左侧车道线不能在右侧  或者 右侧车道线不能在左侧？？？
            # 当车道线的长度很长，那么是否就可以100%确定是车道线而不是误检
            # 通过100%确定的车道线是否就可以删除其他不合理的车道线
            if left_flag:
                left_curve = curve()
                left_curve.exist = True
                left_curve.p1 = p1
                left_curve.p2 = p2 
                left_curve.p3 = p3
                left_curve.p4 = p4
                left_curve.intersect = intersect
                left_curve.angle = angle
                left_curve.length = approx_list.length[i]
                left_curve.curveType = 0

                # append
                left_curves.append(left_curve)
                
            elif right_flag:
                right_curve = curve()
                right_curve.exist = True
                right_curve.p1 = p1
                right_curve.p2 = p2
                right_curve.p3 = p3
                right_curve.p4 = p4
                right_curve.intersect = intersect
                right_curve.angle = angle
                right_curve.length = approx_list.length[i]
                right_curve.curveType = 0

                # append 
                right_curves.append(right_curve)

        elif len(approx_curve) == 3: 
            ps1 = approx_curve[0]
            ps2 = approx_curve[1]
            ps3 = approx_curve[2]

            k1, b1 = cal_line_coef(ps1, ps2)
            step1 = (ps1[0] - ps2[0]) / 16.0
            p1 = ps1
            x_3_4 = ps1[0] - step1 * 14
            p2 = [x_3_4, k1 * x_3_4 + b1]

            k2, b2 = cal_line_coef(ps2, ps3)
            step2 = (ps2[0] - ps3[0]) / 16.0
            x_1_4 = ps2[0] - step2 * 4
            p3 = [x_1_4, k2 * x_1_4 + b2]

            # 两条直线之间的夹角
            angle = np.arctan(abs((k1-k2)/(1+k1*k2))) * 180 / np.pi if k1*k2 != -1 else 90
            # 计算p1 p2平行的直线，距离为d
            k3, b3 = cal_line_coef(p2, p3)
            d = getTransD(angle)

            # d = 10 if angle < 30 else 20
            c1 = d * math.sqrt(k3**2 + 1) - k3 * p3[0] + p3[1]
            c2 = -1 * d * math.sqrt(k3**2 + 1) - k3 * p3[0] + p3[1]
            
            if ps3[0] < ps2[0]: # 往左偏
                c = max(c1, c2) if k3 < 0 else min(c1, c2)
            else: # 往右偏
                c = min(c1, c2) if k3 < 0 else max(c1, c2)
            # c = c1 if c1 < c2 else c2
            # y = k3 * x + c
            # 计算经过点p2、 p3法线
            k_normal_1, b_normal_1 = calNormalLine(k3, p2)
            k_normal_2, b_normal_2 = calNormalLine(k3, p3)

            # 计算法线与平移线的 交点
            p2_x_new = (b_normal_1 - c) / (k3 - k_normal_1 + 0.00001)
            p2_y_new = k3 * p2_x_new + c 

            p3_x_new = (b_normal_2 - c) / (k3 - k_normal_2 + 0.00001)
            p3_y_new = k3 * p3_x_new + c 

            p2 = [p2_x_new, p2_y_new]
            p3 = [p3_x_new, p3_y_new]

            p4 = approx_curve[2]

            intersect = (360 - b1) / k1  # 与x轴的交点
            angle = np.arctan(k1) * 180 / np.pi
            if k1 < 0 and intersect < (middle_x + 30): # left
                # color = (0, 255, 255)
                left_flag = True
                right_flag = False
            elif k1 > 0 and intersect >= (middle_x - 30):  # right
                # color = (255, 0, 255)
                left_flag = False
                right_flag = True
            else:
                left_flag = False
                right_flag = False

            if left_flag:
                left_curve = curve()
                left_curve.exist = True
                left_curve.p1 = p1
                left_curve.p2 = p2 
                left_curve.p3 = p3
                left_curve.p4 = p4
                left_curve.intersect = intersect
                left_curve.angle = angle
                left_curve.length = approx_list.length[i]
                left_curve.curveType = 0 if d == 0 else 1

                # append
                left_curves.append(left_curve)
                
            elif right_flag:
                right_curve = curve()
                right_curve.exist = True
                right_curve.p1 = p1
                right_curve.p2 = p2
                right_curve.p3 = p3
                right_curve.p4 = p4
                right_curve.intersect = intersect
                right_curve.angle = angle
                right_curve.length = approx_list.length[i]
                right_curve.curveType = 0 if d == 0 else 1
                
                # append
                right_curves.append(right_curve)

        # 4个点
        else:
            # 是否需要排序
            p1 = approx_curve[0]
            p2 = approx_curve[1]
            p3 = approx_curve[2]
            p4 = approx_curve[-1]

            # 计算靠近车子的直线斜率以及与x轴的交点
            k_p1p2, b_p1p2 = cal_line_coef(p1, p2)
            angle_slope = np.arctan(k_p1p2) * 180 / np.pi
            intersect = (360 - b_p1p2) / k_p1p2  # 与x轴的交点
            if k_p1p2 < 0 and intersect < (middle_x + 30): # left
                # color = (0, 255, 255)
                left_flag = True
                right_flag = False
            elif k_p1p2 > 0 and intersect > (middle_x - 30):  # right
                # color = (255, 0, 255)
                left_flag = False
                right_flag = True
            else:
                left_flag = False
                right_flag = False

            # 计算两边端点的直线
            # k_p1p2, b_p1p2 = cal_line_coef(p1, p2)
            k_p4p3, b_p4p3 = cal_line_coef(p4, p3)

            # 两条直线之间的夹角
            angle = np.arctan(abs((k_p1p2-k_p4p3)/(1+k_p1p2*k_p4p3))) * 180 / np.pi if k_p1p2*k_p4p3 != -1 else 90

            # 计算p2 p3平行的直线，y = k3 * x + c, 距离为d
            k3, b3 = cal_line_coef(p2, p3)
            # d = 5 if angle < 30 else 20
            d = getTransD(angle)
            c1 = d * math.sqrt(k3**2 + 1) - k3 * p3[0] + p3[1]
            c2 = -1 * d * math.sqrt(k3**2 + 1) - k3 * p3[0] + p3[1]
            if p3[0] < p2[0]: # 往左偏
                c = max(c1, c2) if k3 < 0 else min(c1, c2)
            else: # 往右偏
                c = min(c1, c2) if k3 < 0 else max(c1, c2)

            # 计算经过点p2、 p3法线
            k_normal_1, b_normal_1 = calNormalLine(k3, p2)
            k_normal_2, b_normal_2 = calNormalLine(k3, p3)

            # 计算法线与平移线的 交点
            p2_x_new = (b_normal_1 - c) / (k3 - k_normal_1 + 0.00001)
            p2_y_new = k3 * p2_x_new + c 

            p3_x_new = (b_normal_2 - c) / (k3 - k_normal_2 + 0.00001)
            p3_y_new = k3 * p2_x_new + c 

            p2 = [p2_x_new, p2_y_new]
            p3 = [p3_x_new, p3_y_new]

            if left_flag:
                left_curve = curve()
                left_curve.exist = True
                left_curve.p1 = p1
                left_curve.p2 = p2 
                left_curve.p3 = p3
                left_curve.p4 = p4
                left_curve.intersect = intersect
                left_curve.angle = angle_slope
                left_curve.length = approx_list.length[i]
                left_curve.curveType = 1
                
                # append
                left_curves.append(left_curve)

            elif right_flag:
                right_curve = curve()
                right_curve.exist = True
                right_curve.p1 = p1
                right_curve.p2 = p2
                right_curve.p3 = p3
                right_curve.p4 = p4
                right_curve.intersect = intersect
                right_curve.angle = angle_slope
                right_curve.length = approx_list.length[i]
                right_curve.curveType = 1

                # append
                right_curves.append(right_curve)
                
    return left_curves, right_curves


def trackLane(left_curve, right_curve, left_curves, right_curves, sim_angle=2, abs_intersect=20, length_ratio=1.5, max_gap=25):
    
    # 左侧车道线存在
    if left_curve.exist:
        # 寻找离车道线最近的curve
        
        find_flag = False
        index = -1
        min_gap = 1000
        # 计算原有curve  p1, p2的直线
        k12, b12 = cal_line_coef(left_curve.p1, left_curve.p2)
        for i , left in enumerate(left_curves):
            # 计算p2点到直线的距离
            d = abs(k12 * left.p2[0] - left.p2[1] + b12) / math.sqrt(k12**2 + 1)
            if d < max_gap and d < min_gap:
                min_gap = d
                index = i
                find_flag = True

        if find_flag:
            # 存在且角度不能跳变
            left_curve.birth += 1
            left_curve.die = 0
            if abs(left_curve.angle - left_curves[index].angle) < sim_angle:
                # update
                left_curve.exist = True
                left_curve.p1 = left_curves[index].p1
                left_curve.p2 = left_curves[index].p2
                left_curve.p3 = left_curves[index].p3
                left_curve.p4 = left_curves[index].p4
                left_curve.intersect = left_curves[index].intersect
                left_curve.angle = left_curves[index].angle
                left_curve.length = left_curves[index].length
                left_curve.curveType = left_curves[index].curveType
        else:
            left_curve.die += 1
            left_curve.birth += 1

                

    else: # 不存在

        left_curve.birth = 1
        left_curve.die = 0
        if len(left_curves) == 1:
            left_curve.exist = True
            left_curve.p1 = left_curves[0].p1
            left_curve.p2 = left_curves[0].p2
            left_curve.p3 = left_curves[0].p3
            left_curve.p4 = left_curves[0].p4
            left_curve.intersect = left_curves[0].intersect
            left_curve.angle = left_curves[0].angle
            left_curve.length = left_curves[0].length
            left_curve.curveType = left_curves[0].curveType

        elif len(left_curves) > 1:
            left_curve.exist = True
            left_curve.p1 = left_curves[0].p1
            left_curve.p2 = left_curves[0].p2
            left_curve.p3 = left_curves[0].p3
            left_curve.p4 = left_curves[0].p4
            left_curve.intersect = left_curves[0].intersect
            left_curve.angle = left_curves[0].angle
            left_curve.length = left_curves[0].length
            left_curve.curveType = left_curves[0].curveType

            for i in range(1, len(left_curves)):
                # 直线的情况，需要判断是否需要合并
                merge_flag = False
                if (left_curve.curveType == 0 and left_curves[i].curveType == 0):
                    if left_curves[i].p1[1] > left_curve.p1[1]: # Y值较大
                        k_new, b_new = cal_line_coef(left_curves[i].p1, left_curve.p4)
                        angle_new = np.arctan(k_new) * 180 / np.pi
                        if abs(left_curves[i].angle - left_curve.angle) < sim_angle and abs(angle_new - left_curves[i].angle) < sim_angle and abs(angle_new - left_curve.angle) < sim_angle:
                            left_curve.p1 = left_curves[i].p1
                            left_curve.length = cal_length(left_curve.p1, left_curve.p4)
                            left_curve.angle = angle_new
                            merge_flag = True
                    
                    if left_curves[i].p4[1] < left_curve.p4[1]:
                        k_new, b_new = cal_line_coef(left_curve.p1, left_curves[i].p4)
                        angle_new = np.arctan(k_new) * 180 / np.pi
                        if abs(left_curves[i].angle - left_curve.angle) < sim_angle and abs(angle_new - left_curves[i].angle) < sim_angle and abs(angle_new - left_curve.angle) < sim_angle:
                            left_curve.p4 = left_curves[i].p4
                            left_curve.length = cal_length(left_curve.p1, left_curve.p4)
                            left_curve.angle = angle_new
                            merge_flag = True

                # 如果没有合并，则判断是否满足条件
                if not merge_flag and \
                    ((abs(left_curves[i].intersect - left_curve.intersect) < abs_intersect and left_curves[i].length > left_curve.length * length_ratio) or \
                        (left_curves[i].intersect > left_curve.intersect and left_curves[i].length > length_ratio * left_curve.length)\
                            or ((left_curves[i].intersect - left_curve.intersect) > 45)):
                    # left_curve.exist = True
                    left_curve.p1 = left_curves[i].p1
                    left_curve.p2 = left_curves[i].p2
                    left_curve.p3 = left_curves[i].p3
                    left_curve.p4 = left_curves[i].p4
                    left_curve.intersect = left_curves[i].intersect
                    left_curve.angle = left_curves[i].angle
                    left_curve.length = left_curves[i].length
                    left_curve.curveType = left_curves[i].curveType

    # 右侧车道线存在
    if right_curve.exist:
        # 寻找离车道线最近的curve
        
        find_flag = False
        index = -1
        min_gap = 1000
        # 计算原有curve  p1, p2的直线
        k12, b12 = cal_line_coef(right_curve.p1, right_curve.p2)
        for i , right in enumerate(right_curves):
            # 计算p2点到直线的距离
            d = abs(k12 * right.p2[0] - right.p2[1] + b12) / math.sqrt(k12**2 + 1)
            if d < max_gap and d < min_gap:
                min_gap = d
                index = i
                find_flag = True

        if find_flag:
            right_curve.birth += 1
            right_curve.die = 0
            # 存在且角度不能跳变
            if abs(right_curve.angle - right_curves[index].angle) < sim_angle:
                # update
                right_curve.exist = True
                right_curve.p1 = right_curves[index].p1
                right_curve.p2 = right_curves[index].p2
                right_curve.p3 = right_curves[index].p3
                right_curve.p4 = right_curves[index].p4
                right_curve.intersect = right_curves[index].intersect
                right_curve.angle = right_curves[index].angle
                right_curve.length = right_curves[index].length
                right_curve.curveType = right_curves[index].curveType
        else:
            right_curve.birth += 1
            right_curve.die += 1

    else: # 不存在
        right_curve.birth = 1
        right_curve.die = 0
        if len(right_curves) == 1:  # 直接赋值
            right_curve.exist = True
            right_curve.p1 = right_curves[0].p1
            right_curve.p2 = right_curves[0].p2
            right_curve.p3 = right_curves[0].p3
            right_curve.p4 = right_curves[0].p4
            right_curve.intersect = right_curves[0].intersect
            right_curve.angle = right_curves[0].angle
            right_curve.length = right_curves[0].length
            right_curve.curveType = right_curves[0].curveType
        elif len(right_curves) > 1:
            right_curve.exist = True
            right_curve.p1 = right_curves[0].p1
            right_curve.p2 = right_curves[0].p2
            right_curve.p3 = right_curves[0].p3
            right_curve.p4 = right_curves[0].p4
            right_curve.intersect = right_curves[0].intersect
            right_curve.angle = right_curves[0].angle
            right_curve.length = right_curves[0].length
            right_curve.curveType = right_curves[0].curveType
            for i in range(1, len(right_curves)):
                # 直线的情况，需要判断是否需要合并
                merge_flag = False
                if (right_curve.curveType == 0 and right_curves[i].curveType == 0):
                    if right_curves[i].p1[1] > right_curve.p1[1]: # Y值较大
                        k_new, b_new = cal_line_coef(right_curves[i].p1, right_curve.p4)
                        angle_new = np.arctan(k_new) * 180 / np.pi
                        if abs(right_curves[i].angle - right_curve.angle) < sim_angle and abs(angle_new - right_curves[i].angle) < sim_angle \
                                and abs(angle_new - right_curve.angle) < sim_angle:
                            right_curve.p1 = right_curves[i].p1
                            right_curve.length = cal_length(right_curve.p1, right_curve.p4)
                            right_curve.angle = angle_new
                            merge_flag = True
                    
                    if right_curves[i].p4[1] < right_curve.p4[1]:
                        k_new, b_new = cal_line_coef(right_curve.p1, right_curves[i].p4)
                        angle_new = np.arctan(k_new) * 180 / np.pi
                        if abs(right_curves[i].angle - right_curve.angle) < sim_angle and abs(angle_new - right_curves[i].angle) < sim_angle \
                                and abs(angle_new - right_curve.angle) < sim_angle:
                            right_curve.p4 = right_curves[i].p4
                            right_curve.length = cal_length(right_curve.p1, right_curve.p4)
                            right_curve.angle = angle_new
                            merge_flag = True

                # 如果没有合并，则判断是否满足条件
                if not merge_flag and \
                    ((abs(right_curves[i].intersect - right_curve.intersect) < abs_intersect and right_curves[i].length > right_curve.length * length_ratio) or \
                        (right_curves[i].intersect > right_curve.intersect and right_curves[i].length > length_ratio * right_curve.length)\
                           or ((right_curve.intersect - right_curves[i].intersect) > 45) ):
                    # right_curve.exist = True
                    right_curve.p1 = right_curves[i].p1
                    right_curve.p2 = right_curves[i].p2
                    right_curve.p3 = right_curves[i].p3
                    right_curve.p4 = right_curves[i].p4
                    right_curve.intersect = right_curves[i].intersect
                    right_curve.angle = right_curves[i].angle
                    right_curve.length = right_curves[i].length
                    right_curve.curveType = right_curves[i].curveType

    return 


def fitBezier3_track(img_draw, ll_pts, left_curve=curve(), right_curve=curve(), ratio=1.0, middle_x=320):
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
    length_thres = 77
    length_ratio =  1.2
    length_max_thresh = 180
    sim_angle = 3  # 6
    left_curves = []
    right_curves = []
    max_gap = 25
    abs_intersect = 20
    for n, pts in enumerate(ll_pts):
        approx = cv2.approxPolyDP(pts, 5, True)
        if approx.shape[0] < 2:
            continue

        approx = nms_pt(approx, 5)

        approx_list = approxDP()
        approx_single = []
        process = False

        if approx.shape[0] == 3: # L 型情况
            k01, b01 = cal_line_coef(approx[0][0].reshape(2).tolist(), approx[1][0].reshape(2).tolist())
            k02, b02 = cal_line_coef(approx[0][0].reshape(2).tolist(), approx[2][0].reshape(2).tolist())
            # 计算两条直线的夹角
            angle = np.arctan(abs((k01-k02)/(1+k01*k02))) * 180 / np.pi if k01*k02 != -1 else 90
            if abs(angle - 90) < 6 or abs(angle - 45) < 6:
                process = True
                # 选取长的线作为最终拟合直线
                length01 = cal_length(approx[0][0].reshape(2).tolist(), approx[1][0].reshape(2).tolist())
                length02 = cal_length(approx[0][0].reshape(2).tolist(), approx[2][0].reshape(2).tolist())
                approx_single = []
                if length01 > length02:
                    approx_single.append(approx[0][0].reshape(2).tolist())
                    approx_single.append(approx[1][0].reshape(2).tolist())
                    approx_list.points.append(approx_single)
                    approx_list.length.append(length01)
                else:
                    approx_single.append(approx[0][0].reshape(2).tolist())
                    approx_single.append(approx[2][0].reshape(2).tolist())
                    approx_list.points.append(approx_single)
                    approx_list.length.append(length02)
                

        if approx.shape[0] == 4:  # 处理T字形情况
            k03, b03 = cal_line_coef(approx[0][0].reshape(2).tolist(), approx[3][0].reshape(2).tolist())
            angle03 = abs(np.arctan(k03) * 180 / np.pi)
            if angle03 < 15:
                T_flag = False
            else:
                T_flag = True
            if T_flag:
                k01, b01 = cal_line_coef(approx[0][0].reshape(2).tolist(), approx[1][0].reshape(2).tolist())
                k31, b31 = cal_line_coef(approx[3][0].reshape(2).tolist(), approx[1][0].reshape(2).tolist())
                angle01 = np.arctan(k01) * 180 / np.pi
                angle31 = np.arctan(k31) * 180 / np.pi
                if abs(angle01 - angle31) < 5:
                    approx_single = []
                    approx_single.append(approx[0][0].reshape(2).tolist())
                    approx_single.append(approx[1][0].reshape(2).tolist())
                    approx_single.append(approx[3][0].reshape(2).tolist())
                    approx_list.points.append(approx_single)
                    length1 = cal_length(approx[0][0].reshape(2).tolist(), approx[1][0].reshape(2).tolist())
                    length2 = cal_length(approx[3][0].reshape(2).tolist(), approx[1][0].reshape(2).tolist())
                    approx_list.length.append(length1+length2)
                    process = True
                
                if not process:
                    k02, b02 = cal_line_coef(approx[0][0].reshape(2).tolist(), approx[2][0].reshape(2).tolist())
                    k32, b32 = cal_line_coef(approx[3][0].reshape(2).tolist(), approx[2][0].reshape(2).tolist())
                    angle02 = np.arctan(k02) * 180 / np.pi
                    angle32 = np.arctan(k32) * 180 / np.pi
                    if abs(angle02 - angle32) < 5:
                        approx_single = []
                        approx_single.append(approx[0][0].reshape(2).tolist())
                        approx_single.append(approx[2][0].reshape(2).tolist())
                        approx_single.append(approx[3][0].reshape(2).tolist())
                        approx_list.points.append(approx_single)
                        length1 = cal_length(approx[0][0].reshape(2).tolist(), approx[2][0].reshape(2).tolist())
                        length2 = cal_length(approx[3][0].reshape(2).tolist(), approx[2][0].reshape(2).tolist())
                        approx_list.length.append(length1+length2)
                        process = True

        if not process:
        # else:    
            # 对多边形拟合后的点进行处理
            
            if approx.shape[0] == 3:
                approx = casePt3(approx.reshape(approx.shape[0], 2).tolist())
            
            total_length = 0.0 
            start = True
            for i in range(approx.shape[0]-1):
                # 过滤斜率较小的直线
                k, b = cal_line_coef(approx[i][0].tolist(), approx[i+1][0].tolist())
                angle = np.arctan(k) * 180 / np.pi
                if abs(angle) < 15:
                    if len(approx_single) > 0:
                        approx_list.points.append(approx_single)
                        approx_list.length.append(total_length)
                        approx_single = []
                        total_length = 0.0
                    start = True
                    continue
                if start:
                    approx_single.append(approx[i][0].tolist())
                    approx_single.append(approx[i+1][0].tolist())
                    total_length += cal_length(approx[i][0].tolist(), approx[i+1][0].tolist())
                    start = False
                else:
                    approx_single.append(approx[i+1][0].tolist())
                    total_length += cal_length(approx[i][0].tolist(), approx[i+1][0].tolist())

            if len(approx_single) > 0:
                approx_list.points.append(approx_single)
                approx_list.length.append(total_length)

        # get left_curves list and right_curves list
        sub_left_curves, sub_right_curves = getLeftRightCurves(approx_list,
                                            length_thres=length_thres, middle_x=middle_x)
        
        if sub_left_curves is not None:
            left_curves += sub_left_curves
        if sub_right_curves is not None:
            right_curves += sub_right_curves
    
    # track
    trackLane(left_curve, right_curve, left_curves, right_curves, sim_angle=sim_angle, abs_intersect=abs_intersect,
                length_ratio=length_ratio, max_gap=max_gap)
    
    # update
    if left_curve.die > 2:
        # left_curve = curve()
        left_curve.exist = False
        left_curve.p1 = None
        left_curve.p2 = None
        left_curve.p3 = None
        left_curve.p4 = None
        left_curve.p5 = None
        left_curve.intersect = 0.0
        left_curve.angle = 0.0
        left_curve.length = 0.0
        left_curve.curveType = -1
        left_curve.birth = 0
        left_curve.die = 0
    if right_curve.die > 2:
        # right_curve = curve()
        right_curve.exist = False
        right_curve.p1 = None
        right_curve.p2 = None
        right_curve.p3 = None
        right_curve.p4 = None
        right_curve.p5 = None
        right_curve.intersect = 0.0
        right_curve.angle = 0.0
        right_curve.length = 0.0
        right_curve.curveType = -1
        right_curve.birth = 0
        right_curve.die = 0

    # draw
    if left_curve.exist:
        T = np.linspace(0, 1, 200)
        for t in T:
            x = int(cal_bezier3(t, left_curve.p1[0], left_curve.p2[0], left_curve.p3[0], left_curve.p4[0]) / ratio)
            y = int(cal_bezier3(t, left_curve.p1[1], left_curve.p2[1], left_curve.p3[1], left_curve.p4[1]) / ratio)

            cv2.circle(img_draw, (x, y), 3, (255, 0, 255), -1)
        
        cv2.circle(img_draw, tuple([int(e / ratio) for e in left_curve.p1]), 6, (0, 0, 255), -1)
        # cv2.circle(img_draw, tuple([int(e / ratio) for e in left_curve.p2]), 6, (0, 0, 255), -1)
        # cv2.circle(img_draw, tuple([int(e / ratio) for e in left_curve.p3]), 6, (0, 0, 255), -1)
        cv2.circle(img_draw, tuple([int(e / ratio) for e in left_curve.p4]), 6, (0, 0, 255), -1)

        cv2.putText(img_draw, f"left curve angle: {abs(left_curve.angle):.1f}", (20, 30), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 0, 255), 2)
    else:
        cv2.putText(img_draw, f"left curve angle: None", (20, 30), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 0, 255), 2)
    
    if right_curve.exist:
        T = np.linspace(0, 1, 200)
        for t in T:
            x = int(cal_bezier3(t, right_curve.p1[0], right_curve.p2[0], right_curve.p3[0], right_curve.p4[0]) / ratio)
            y = int(cal_bezier3(t, right_curve.p1[1], right_curve.p2[1], right_curve.p3[1], right_curve.p4[1]) / ratio)

            cv2.circle(img_draw, (x, y), 3, (0, 255, 255), -1)
        
        cv2.circle(img_draw, tuple([int(e / ratio) for e in right_curve.p1]), 6, (0, 0, 255), -1)
        # cv2.circle(img_draw, tuple([int(e / ratio) for e in right_curve.p2]), 6, (0, 0, 255), -1)
        # cv2.circle(img_draw, tuple([int(e / ratio) for e in right_curve.p3]), 6, (0, 0, 255), -1)
        cv2.circle(img_draw, tuple([int(e / ratio) for e in right_curve.p4]), 6, (0, 0, 255), -1)

        cv2.putText(img_draw, f"right curve angle: {abs(right_curve.angle):.1f}", (20, 60), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 0, 255), 2)
    else:
        cv2.putText(img_draw, f"right curve angle: None", (20, 60), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 0, 255), 2)
    return img_draw



def fitPolyDP(img_draw, ll_pts, ratio=1.0):
    # print("多边形拟合...")
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]

    length_thres = 100
    left_curve = curve()
    right_curve = curve()

    for i, pts in enumerate(ll_pts):
        approx = cv2.approxPolyDP(pts, 10, False)
        # print(f"approx: f{approx.shape}")

        if approx.shape[0] < 2:
            continue

        # 过滤斜率较小的直线
        if approx.shape[0] == 2:
            k, b = cal_line_coef(approx[0][0].tolist(), approx[1][0].tolist())
            angle = np.arctan(k) * 180 / np.pi
            if abs(angle) < 15:
                continue

            if(math.sqrt(math.pow(approx[0][0][0] - approx[1][0][0], 2) + math.pow(approx[0][0][1] - approx[1][0][1], 2)) < 50.0):
                continue

        if approx.shape[0] > 3:
            approx = approx[-3:, ...]  # 最多取两条线
        
        # 只有一条直线
        if approx.shape[0] == 2:
            p1, p2 = (approx[0][0].tolist(), approx[1][0].tolist()) \
                if approx[0][0][1] > approx[1][0][1] else \
                    (approx[1][0].tolist(), approx[0][0].tolist())

            k, b = cal_line_coef(p1, p2)
            intersect = (360 - b) / k  # 与x轴的交点

            if k < 0: # left
                # color = (0, 255, 255)
                left_flag = True
            else:  # right
                # color = (255, 0, 255)
                left_flag = False
            
            # p1 = tuple([int(p1[0] / ratio), int(p1[1] / ratio)])
            # p2 = tuple([int(p2[0] / ratio), int(p2[1] / ratio)])
            # cv2.line(img_draw, tuple(p1), tuple(p2), color, 6)
            # cv2.circle(img_draw, tuple(p1), 8, (0, 0, 255), -1)
            # cv2.circle(img_draw, tuple(p2), 8, (0, 0, 255), -1)

            length = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            if left_flag:
                if not left_curve.exist:  # 第一条曲线
                    left_curve.exist = True
                    left_curve.p1 = p1
                    left_curve.p2 = p2
                    left_curve.intersect = intersect
                else:
                    if (intersect > left_curve.intersect) and length >= length_thres:
                        # left_curve.exist = True
                        left_curve.p1 = p1
                        left_curve.p2 = p2
                        left_curve.intersect = intersect
            else:
                if not right_curve.exist:  # 第一条曲线
                    right_curve.exist = True
                    right_curve.p1 = p1
                    right_curve.p2 = p2
                    right_curve.intersect = intersect
                else:
                    # 如果车道线更靠近车子
                    if (intersect < right_curve.intersect) and length >= length_thres:  
                        # left_curve.exist = True
                        right_curve.p1 = p1
                        right_curve.p2 = p2
                        right_curve.intersect = intersect


        elif approx.shape[0] == 3:

            # 根据Y值确定起止坐标，越靠近车子，Y值越大
            p1, p2, p3 = (approx[0][0].tolist(), approx[1][0].tolist(), approx[2][0].tolist()) \
                if approx[0][0][1] > approx[2][0][1] \
                    else (approx[2][0].tolist(), approx[1][0].tolist(), approx[0][0].tolist())
            k, b = cal_line_coef(p1, p2)
            if k < 0:
                # color = (0, 255, 255)
                left_flag = True
            else:
                # color = (255, 0, 255)
                left_flag = False

            intersect = (360 - b) / k
            length = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) + \
                    math.sqrt((p3[0] - p2[0])**2 + (p3[1] - p2[1])**2)

            if left_flag:
                if not left_curve.exist:  # 第一条曲线
                    left_curve.exist = True
                    left_curve.p1 = p1
                    left_curve.p2 = p2
                    left_curve.p3 = p3
                    left_curve.intersect = intersect
                else:
                    if (intersect > left_curve.intersect) and length >= length_thres:
                        # left_curve.exist = True
                        left_curve.p1 = p1
                        left_curve.p2 = p2
                        left_curve.p3 = p3
                        left_curve.intersect = intersect
            else:
                if not right_curve.exist:  # 第一条曲线
                    right_curve.exist = True
                    right_curve.p1 = p1
                    right_curve.p2 = p2
                    right_curve.p3 = p3
                    right_curve.intersect = intersect
                else:
                    # 如果车道线更靠近车子
                    if (intersect < right_curve.intersect) and length_thres >= length_thres:  
                        # left_curve.exist = True
                        right_curve.p1 = p1
                        right_curve.p2 = p2
                        right_curve.p3 = p3
                        right_curve.intersect = intersect

            # p1 = tuple([int(p1[0] / ratio), int(p1[1] / ratio)])
            # p2 = tuple([int(p2[0] / ratio), int(p2[1] / ratio)])
            # p3 = tuple([int(p3[0] / ratio), int(p3[1] / ratio)])
            # cv2.line(img_draw, tuple(p1), tuple(p2), color, 6)
            # cv2.line(img_draw, tuple(p2), tuple(p3), color, 6)
            # cv2.circle(img_draw, tuple(p1), 8, (0, 0, 255), -1)
            # cv2.circle(img_draw, tuple(p2), 8, (0, 0, 255), -1)
            # cv2.circle(img_draw, tuple(p3), 8, (0, 0, 255), -1)

    if left_curve.exist:
        color = (255, 0, 255)
        left_curve.p1 = tuple([int(left_curve.p1[0] / ratio), int(left_curve.p1[1] / ratio)])
        left_curve.p2 = tuple([int(left_curve.p2[0] / ratio), int(left_curve.p2[1] / ratio)])
        cv2.line(img_draw, tuple(left_curve.p1), tuple(left_curve.p2), color, 6)
        cv2.circle(img_draw, tuple(left_curve.p1), 8, (0, 0, 255), -1)
        cv2.circle(img_draw, tuple(left_curve.p2), 8, (0, 0, 255), -1)
        if left_curve.p3 is not None:
            left_curve.p3 = tuple([int(left_curve.p3[0] / ratio), int(left_curve.p3[1] / ratio)])
            cv2.line(img_draw, tuple(left_curve.p3), tuple(left_curve.p2), color, 6)
            cv2.circle(img_draw, tuple(left_curve.p3), 8, (0, 0, 255), -1)

    if right_curve.exist:
        color = (0, 255, 255)
        right_curve.p1 = tuple([int(right_curve.p1[0] / ratio), int(right_curve.p1[1] / ratio)])
        right_curve.p2 = tuple([int(right_curve.p2[0] / ratio), int(right_curve.p2[1] / ratio)])
        cv2.line(img_draw, tuple(right_curve.p1), tuple(right_curve.p2), color, 6)
        cv2.circle(img_draw, tuple(right_curve.p1), 8, (0, 0, 255), -1)
        cv2.circle(img_draw, tuple(right_curve.p2), 8, (0, 0, 255), -1)
        if right_curve.p3 is not None:
            right_curve.p3 = tuple([int(right_curve.p3[0] / ratio), int(right_curve.p3[1] / ratio)])
            cv2.line(img_draw, tuple(right_curve.p3), tuple(right_curve.p2), color, 6)
            cv2.circle(img_draw, tuple(right_curve.p3), 8, (0, 0, 255), -1)
        
    return img_draw


def fitSplines(img_draw, ll_pts, ratio=1.0):
    # print("样条插值拟合...")
    # splines 曲线插值拟合
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
    step = 10

    for pts in ll_pts:
        approx = cv2.approxPolyDP(pts, 10, False)
        if approx.shape[0] < 2:
            continue

        # print(f"approx: f{approx.shape}")
        # 过滤斜率较小的直线
        if approx.shape[0] == 2:
            k, b = cal_line_coef(approx[0][0].tolist(), approx[1][0].tolist())
            angle = np.arctan(k) * 180 / np.pi
            if abs(angle) < 15:
                continue

            if(math.sqrt(math.pow(approx[0][0][0] - approx[1][0][0], 2) + math.pow(approx[0][0][1] - approx[1][0][1], 2)) < 50.0):
                continue

        X, Y = [], []
        for p in pts:
            X.append(p[0][0])
            Y.append(p[0][1])

        if (len(X) < 2 * step):
            # print(f"points too less, {len(X)}")
            continue

        try:
            r_x, r_y, r_yaw, r_k, travel = calc_2d_spline_interpolation(X[::step], Y[::step], num=100)  

            # color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            points_fitted = []
            for j, ps in enumerate(zip(r_x, r_y)):
                if np.isnan(ps[0]) or np.isnan(ps[1]):
                    continue
                p = tuple([int(s / ratio) for s in ps])
                cv2.circle(img_draw, p, 4, (0, 255, 255), -1)
                # points_fitted.append(p)
                # if j > 0:
                #     cv2.line(img_re, points_fitted[-1], points_fitted[-2], colors[i % len(colors)], 3)

            for xy in zip(X[::step], Y[::step]):
                xy = tuple([int(t / ratio) for t in xy])
                cv2.circle(img_draw, xy, 8, (0, 0, 255), -1)
        except Exception as e:
            # print(e)
            continue
            # sys.exit()

    return img_draw


if __name__ == "__main__":
    # img = cv2.imread("inference/output/ll_output_net.png", cv2.IMREAD_GRAYSCALE)
    # img = cv2.medianBlur(img, 5)
    # cv2.imwrite("./inference/output/ll_blur.png", img)
    # ll_pts = FindPts(img)

    imgRootPath = "/data/linkejun/dataset/sourceImg/20230901154959"
    imgPaths = getFilePaths(imgRootPath)
    print(f"total image path: {len(imgPaths)}")
