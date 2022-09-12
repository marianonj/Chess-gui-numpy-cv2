import time as time_method

import numpy as np
import cv2
import timeit
import sounddevice as sd
import win32api
from win32con import VK_MEDIA_PLAY_PAUSE, KEYEVENTF_EXTENDEDKEY

state = 0  # off

color_gradient = np.column_stack((np.hstack((np.arange(0, 256), np.full(255, 255), np.full(255, 255), np.arange(0, 256)[::-1], np.full(255, 0), np.full(255, 0))),
                                  np.hstack((np.full(255, 0), np.full(255, 0), np.arange(0, 256), np.full(255, 255), np.full(255, 255), np.arange(0, 256)[::-1])),
                                  np.hstack((np.full(255, 255), np.arange(0, 256)[::-1], np.full(255, 0), np.full(255, 0), np.arange(0, 256), np.full(255, 255))))).astype(np.uint8)


def save_video(image_list, fps, color, output_str):
    height, width = image_list[0].shape[0], image_list[0].shape[1]
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    if color:
        out = cv2.VideoWriter(f'{output_str}.avi', fourcc, fps, (width, height))
    else:
        out = cv2.VideoWriter(f'{output_str}.avi', fourcc, fps, (width, height), 0)
    for image in image_list:
        out.write(image)
    out.release()


def image_show(img, resize=None):
    if resize:
        img_resize = cv2.resize(img, (int(img.shape[1] * resize), int(img.shape[0] * resize)))
        cv2.imshow('test_reads', img_resize)
    else:
        cv2.imshow('test_reads', img)

    cv2.moveWindow('test_reads', 0, 0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def time_methods(*methods, setup='Pass', number=1000000):
    for count, method in enumerate(methods):
        print(f'Method {count} takes {timeit.timeit(method, setup=setup, number=number)}')


def v_range(starts, stops):
    length_range = stops - starts
    return np.repeat(stops - length_range.cumsum(), length_range) + np.arange(length_range.sum())


def calculate_bluetooth_delay():
    offset_list = []
    timings_list = []

    def calcluate_base_offset(indata, outdata, frames, time, status):
        offset_list.append(np.linalg.norm(indata) * 10)

    with sd.Stream(callback=calcluate_base_offset):
        sd.sleep(10000)

    offset_value = sum(offset_list) / len(offset_list)

    def calculate_spotify_bluetooth_delay(indata, outdata, frames, time, status):
        time_method.sleep(2)
        global state
        if state == 0:
            if (np.linalg.norm(indata) * 10) - offset_value < 30:
                timings_list.append(time_method.perf_counter())
                win32api.keybd_event(VK_MEDIA_PLAY_PAUSE, 0, KEYEVENTF_EXTENDEDKEY, 0)
                state = 1
        if state == 1:
            if (np.linalg.norm(indata) * 10) - offset_value > 50:
                timings_list.append(time_method.perf_counter())
                win32api.keybd_event(VK_MEDIA_PLAY_PAUSE, 0, KEYEVENTF_EXTENDEDKEY, 0)
                state = 0

    with sd.Stream(callback=calculate_spotify_bluetooth_delay):
        sd.sleep(100000)
    timings = np.array(timings_list)
    timing_diffs = np.diff(timings)
    average_delay = np.mean(np.abs(timings))
    print('b')

def call_all_local_functions(local_funcs):
    for key, func in local_funcs:
        if callable(func):
            func()

def box_overlap_2d(bbox_1, bbox_2) -> bool:
    if bbox_1[0] > bbox_2[1] > bbox_1[0] and bbox_1[2] > bbox_2[3] > bbox_2[2]:
        return False
    else:
        return True

