import numpy as np


def multithreshold(img, thresholds):
    masks = np.zeros((len(thresholds) + 1, img.shape[0], img.shape[1]), bool)
    for i, t in enumerate(sorted(thresholds)):
        masks[i+1] = (img > t)
    masks[0] = ~masks[1]
    for i in range(1, len(masks) - 1):
        masks[i] = masks[i] ^ masks[i+1]
    return masks


def otsu_method(hist):
    num_bins = hist.shape[0] # hist의 레벨의 수(예: hist[8-32] => 24)
    total = hist.sum() # hist의 전체 픽셀 수
    sum_total = np.dot(range(0, num_bins), hist) # 전체 (레벨 * 레벨에서의 픽셀수)

    weight_background = 0.0
    sum_background = 0.0

    optimum_value = 0
    max_variance = -np.inf

    for t in range(0, num_bins):
        # background weight will be incremented, while foreground's will be reduced
        weight_background += hist.item(t) # t번째까지의 픽셀수
        if weight_background == 0:
            continue

        weight_foreground = total - weight_background # 나머지(전체-t까지) 픽셀수
        if weight_foreground == 0: # t까지의 픽셀수 == 전체 픽셀수 -> 빠져나옴
            break

        sum_background += t * hist.item(t) # t까지의 (레벨 * 레벨의 픽셀수)

        mean_background = sum_background / weight_background  # t까지의 (레벨 * 레벨의 픽셀수)/t까지의 픽셀수
        mean_foreground = (sum_total - sum_background) / weight_foreground # 나머지 (레벨 * 레벨의 픽셀수)/나머지(t~마지막) 픽셀수
        # mean_background: t까지 평균 gray level, mean_foreground: t이후의 평균 gray level
        # 급간 분산
        inter_class_variance = weight_background * weight_foreground * \
            (mean_background - mean_foreground) ** 2
        ##print(inter_class_variance)

        # find the threshold with maximum variances between classes
        # 급간 분산 최대화하는 임계값(optimum_value) 찾기
        if inter_class_variance > max_variance:
            optimum_value = t
            max_variance = inter_class_variance

    return optimum_value, max_variance


def normalised_histogram_binning(hist, M=64, L=256):
    norm_hist = np.zeros((M, 1), dtype=np.float32) # 0으로 초기화
    N = L // M # L: 전체 gray level 수, M: bin group 수, N: 하나의 bin qruop에 묶이는 level 수
    counters = [range(x, x+N) for x in range(0, L, N)] # bin group 수와 동일 (N=8 -> C0[0:8]-C31[248:256])
    for i, C in enumerate(counters):
        norm_hist[i] = 0
        for j in C:
            norm_hist[i] += hist[j] # i번째 counter 안의 값들을 합침 (C0:hist 0-7까지의 합, C1:8-15까지의 합)
    norm_hist = (norm_hist / norm_hist.max()) * 100 # 값을 0-100 사이의 값으로 변환

    for i in range(0, M):

      #  print(norm_hist[i])

        if norm_hist[i] < 0.02:
            norm_hist[i] = 0
    return norm_hist


def find_valleys(H):
    hsize = H.shape[0]
    probs = np.zeros((hsize, 1), dtype=int)
    costs = np.zeros((hsize, 1), dtype=float)
    for i in range(1, hsize-1): # 각 counter에 (valley가 될)확률 할당(0%, 25%, 75%, 100%)
        if H[i] > H[i-1] or H[i] > H[i+1]:
            probs[i] = 0
        elif H[i] < H[i-1] and H[i] == H[i+1]:
            probs[i] = 25
            costs[i] = H[i-1] - H[i]
        elif H[i] == H[i-1] and H[i] < H[i+1]:
            probs[i] = 75
            costs[i] = H[i+1] - H[i]
        elif H[i] < H[i-1] and H[i] < H[i+1]:
            probs[i] = 100
            costs[i] = (H[i-1] + H[i+1]) - 2*H[i]
        elif H[i] == H[i-1] and H[i] == H[i+1]:
            probs[i] = probs[i-1]
            costs[i] = probs[i-1]

    # for i in range(1, hsize-1):
    #     if H[i-1]+H[i]+H[i+1]<100:
    #         probs[i]=0
    #     elif H[i-1]+H[i]+H[i+1]>=100:
    #         probs[i]=100
    #     probs[0]=probs[31]=0

    for i in range(1, hsize-1):
        if probs[i] != 0:
            # floor devision. if > 100 = 1, else 0
            # [i의 확률이 0이 아니면] i-1(전), i, i+1(후) 확률 합이 100%미만=>0%으로
            probs[i] = (probs[i-1] + probs[i] + probs[i+1]) // 100
    valleys = [i for i, x in enumerate(probs) if x > 0] # 100%이하: x(probs[i])==0, 100%이상: x>=1

    # if maximum is not None and maximum < len(valleys):
    # valleys = sorted(valleys, key=lambda x: costs[x])[0:maximum]
    return valleys


def valley_estimation(hist, M=64, L=256):
    # Normalised histogram binning
    norm_hist = normalised_histogram_binning(hist, M, L)
    valleys = find_valleys(norm_hist) # valley 리스트 반환 => valley 수=리스트 length
    return valleys


def threshold_valley_regions(hist, valleys, N):
    thresholds = []
    for valley in valleys:
        start_pos = (valley * N) - N # 해당 valley의 왼쪽꺼부터 (valley=valley에 해당하는 bin group, N=하나의 bin group 안의 gray level 개수)
        end_pos = (valley + 2) * N # valley의 오른쪽꺼까지
        hist_part = hist[start_pos:end_pos] # N=8, valley가 Ci[8-16]이면, start_pos=0(왼쪽 Ci-1[0-8]), end_pos=24(왼쪽 Ci+1[16-24])
        sub_threshold, val = otsu_method(hist_part)
        thresholds.append((start_pos + sub_threshold, val))
    thresholds.sort(key=lambda x: x[1], reverse=True)
    thresholds, values = [list(t) for t in zip(*thresholds)]
    return thresholds


def modified_TSMO(hist, M=128, L=256):
    N = L // M
    valleys = valley_estimation(hist, M, L)
    thresholds = threshold_valley_regions(hist, valleys, N)
    return thresholds
