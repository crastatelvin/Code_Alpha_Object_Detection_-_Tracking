import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

def iou_batch(bb_test, bb_gt):
    """
    Computes IoU between two sets of bounding boxes.
    bb_test: [N, 4] - predicted boxes (x1, y1, x2, y2)
    bb_gt: [M, 4] - detection boxes (x1, y1, x2, y2)
    Returns: [N, M] array of IoU values.
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1]) 
             + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return o

def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h    # scale is area
    r = w / float(h) if h > 0 else 0
    return np.array([x, y, s, r]).reshape((4, 1))

def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the state form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w if w > 0 else 0
    if (x[3] <= 0):
        h = 0
    x1 = x[0] - w/2.
    y1 = x[1] - h/2.
    x2 = x[0] + w/2.
    y2 = x[1] + h/2.
    if score is None:
        return np.array([x1, y1, x2, y2]).reshape((1, 4))
    else:
        return np.array([x1, y1, x2, y2, score]).reshape((1, 5))

class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0
    def __init__(self, bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]
        ])
        self.kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0]
        ])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if ((self.kf.x[6] + self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1][0]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)[0]

def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold)
        if a.all(axis=0).all() == a.all(axis=1).all():
            matched_indices = np.stack(linear_sum_assignment(-iou_matrix), axis=1)
        else:
            matched_indices = np.empty((0, 2), dtype=int)
    else:
        matched_indices = np.empty((0, 2), dtype=int)

    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
            
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matches with low IOU
    matches = []
    for m in matched_indices:
        if (iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
            
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class Sort(object):
    """
    SORT tracker manager.
    """
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5)), frame=None):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],...]
          frame - BGR frame image (ignored in SORT)
        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.delete(trks, to_del, axis=0)
        for t in reversed(to_del):
            self.trackers.pop(t)
            
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets[:, :4], trks[:, :4], self.iou_threshold
        )

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :4])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :4])
            self.trackers.append(trk)
            
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, 5)) # +1 as ID starts from 1
            i -= 1
            # remove dead trackers
            if (trk.time_since_update > self.max_age):
                self.trackers.pop(i)
                
        if (len(ret) > 0):
            return np.concatenate(ret, axis=0)
        return np.empty((0, 5))

from deep_sort_realtime.deepsort_tracker import DeepSort

class DeepSortTracker:
    """
    A wrapper class for Deep SORT tracker using the deep-sort-realtime package.
    """
    def __init__(self, max_age=5, n_init=3, max_cosine_distance=0.2):
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance,
            embedder="mobilenet",
            half=False,
            bgr=True
        )

    def update(self, dets=np.empty((0, 5)), frame=None):
        """
        Updates the tracker with new detections.
        dets: numpy array of [x1, y1, x2, y2, score]
        frame: BGR frame image (required for deep sort appearance features)
        Returns:
            tracks: numpy array of [x1, y1, x2, y2, track_id]
        """
        if frame is None:
            raise ValueError("Deep SORT tracker requires the BGR frame for visual feature extraction.")

        # Convert detections to Deep SORT format: [([x, y, w, h], confidence, class_name)]
        ds_dets = []
        for det in dets:
            x1, y1, x2, y2, score = det
            w = x2 - x1
            h = y2 - y1
            # Deep SORT requires a list/tuple format: ([left, top, width, height], confidence, class_name)
            # We can use a dummy class label like "object" since YOLO class filtering is done beforehand.
            ds_dets.append(([float(x1), float(y1), float(w), float(h)], float(score), "object"))

        # Update tracks
        ds_tracks = self.tracker.update_tracks(ds_dets, frame=frame)

        ret = []
        for track in ds_tracks:
            # We only keep tracks that are confirmed and active
            if not track.is_confirmed():
                continue
            
            # get_ltwh returns [left, top, width, height]
            ltwh = track.to_ltwh()
            x1 = ltwh[0]
            y1 = ltwh[1]
            x2 = x1 + ltwh[2]
            y2 = y1 + ltwh[3]
            
            try:
                track_id = int(track.track_id)
            except ValueError:
                # If track ID is a string, hash it to get an integer ID
                track_id = int(hash(track.track_id) % 10000)

            ret.append([x1, y1, x2, y2, track_id])

        if len(ret) > 0:
            return np.array(ret)
        return np.empty((0, 5))

