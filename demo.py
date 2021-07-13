from __future__ import division, print_function, absolute_import

import copy
import os
import tensorflow as tf
import warnings
import cv2
import numpy as np
from multiprocessing import Process, Manager
import time
from PIL import Image
import operator
from yolo_v4 import YOLO4
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import collections
from reid import REID
from itertools import chain
from collections import defaultdict

from google.cloud import bigquery, storage

yolo = YOLO4()
reid = REID()


def get_frame(i, frame):
    project_id = 'mythic-fire-318606'
    bucket_id = 'mythic-fire-318606.appspot.com'
    dataset_id = 'sanhak_2021'
    table_id = 'video' + str(i)

    storage_client = storage.Client()
    db_client = bigquery.Client()
    bucket = storage_client.bucket(bucket_id)
    select_query = (
        "SELECT datetime, path FROM {}.{}.{} ORDER BY datetime LIMIT 1".format(project_id, dataset_id, table_id))

    query_job = db_client.query(select_query)
    results = query_job.result()
    for row in results:
        path = row.path
        date_time = row.datetime

    delete_query = (
        "DELETE FROM {}.{}.{} WHERE datetime = '{}' LIMIT 1".format(project_id, dataset_id, table_id, date_time))


    cam = cv2.VideoCapture(path)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    start_time = time.time()
    if cam.isOpened():
        while True:
            ret, img = cam.read()
            if ret:
                cv2.waitKey(33)  # what is this??
                frame.append(img)
            else:
                break
    else:
        print('cannot open the vid #' + str(i))
        exit()
    # while True:
    #     ret, realframe = cam.read()
    #     if (time.time() - start_time) >= 3:
    #         cam.release()
    #         break
    #     frame.append(realframe)


def gogo(images_by_id, frames, ids_per_frame):
    max_cosine_distance = 0.5
    nn_budget = None
    nms_max_overlap = 0.3
    frame_nums = (len(frames))

    model_filename = 'model_data/models/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)  # use to get feature

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric, max_age=100)

    width = 640
    height = 480
    frame_cnt = 0
    track_cnt = dict()

    for frame in frames:
        image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
        boxs = yolo.detect_image(image)  # n * [topleft_x, topleft_y, w, h]
        features = encoder(frame, boxs)  # n * 128
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]  # length = n

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.delete_overlap_box(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]  # length = len(indices)

        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        tmp_ids = []
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            bbox = track.to_tlbr()
            area = (int(bbox[2]) - int(bbox[0])) * (int(bbox[3]) - int(bbox[1]))
            if bbox[0] >= 0 and bbox[1] >= 0 and bbox[3] < height and bbox[2] < width:
                tmp_ids.append(track.track_id)
                if track.track_id not in track_cnt:
                    track_cnt[track.track_id] = [
                        [frame_cnt, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), area]]
                    images_by_id[track.track_id] = [frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]]

                else:
                    track_cnt[track.track_id].append(
                        [frame_cnt, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), area])
                    images_by_id[track.track_id].append(frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])])
        ids_per_frame.append(set(tmp_ids))
        frame_cnt += 1
        print(frame_cnt, '/', frame_nums)


def Reid(return_list, return_list2, ids_per_frame1, ids_per_frame2):
    threshold = 320
    exist_ids = set()
    final_fuse_id = dict()
    images_by_id = dict()
    feats = dict()
    ids_per_frame = list()
    length = len(return_list)
    print(return_list)
    print(return_list2)

    for key, value in return_list2.items():
        return_list[key + length] = return_list2[key]
    return_list.update(return_list2)
    images_by_id = return_list

    print(images_by_id)

    for i in ids_per_frame2:
        d = set()
        for k in i:
            k += length
            d.add(k)
        ids_per_frame1.append(d)

    ids_per_frame = copy.deepcopy(ids_per_frame1)

    for i, value in images_by_id.items():
        feats[i] = reid._features(images_by_id[i])  # reid._features(images_by_id[i][:min(len(images_by_id[i]),100)])
    print(ids_per_frame)
    for f in ids_per_frame:
        if f:
            if len(exist_ids) == 0:
                for i in f:
                    final_fuse_id[i] = [i]
                exist_ids = f
            else:
                new_ids = f - exist_ids
                for nid in new_ids:
                    dis = []
                    """
                    if len(images_by_id[nid]) > 5:
                        exist_ids.add(nid)
                        continue
                    """
                    unpickable = []
                    for i in f:
                        for key, item in final_fuse_id.items():
                            if i in item:
                                unpickable += final_fuse_id[key]
                    for oid in (exist_ids - set(unpickable)) & set(final_fuse_id.keys()):
                        tmp = np.mean(reid.compute_distance(feats[nid], feats[oid]))
                        dis.append([oid, tmp])
                    exist_ids.add(nid)
                    if not dis:
                        final_fuse_id[nid] = [nid]
                        continue
                    dis.sort(key=operator.itemgetter(1))
                    if dis[0][1] < threshold:
                        combined_id = dis[0][0]
                        images_by_id[combined_id] += images_by_id[nid]
                        final_fuse_id[combined_id].append(nid)
                    else:
                        final_fuse_id[nid] = [nid]

    people_num = len(final_fuse_id)
    print('Final ids and their sub-ids:', final_fuse_id)
    print(people_num)


warnings.filterwarnings('ignore')

if __name__ == '__main__':
    credential_path = "mythic-fire-318606-5b15a08cba70.json"
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.3)
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

    with Manager() as manager:
        frame_get = manager.list()
        frame_get2 = manager.list()
        ids_per_frame = manager.list()
        ids_per_frame2 = manager.list()
        return_list = manager.dict()
        return_list2 = manager.dict()
        p = Process(target=get_frame, args=(0, frame_get))
        p2 = Process(target=get_frame, args=(1, frame_get2))
        p3 = Process(target=gogo, args=(return_list, frame_get, ids_per_frame))
        p4 = Process(target=gogo, args=(return_list2, frame_get2, ids_per_frame2))
        p.start()
        p2.start()
        p.join()
        p2.join()
        p3.start()
        p4.start()
        p3.join()
        p4.join()
        p5 = Process(target=Reid, args=(return_list, return_list2, ids_per_frame, ids_per_frame2))
        p5.start()
        p5.join()

