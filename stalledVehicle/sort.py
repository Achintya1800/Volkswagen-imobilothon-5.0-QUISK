# SORT: Simple Online and Realtime Tracking (Enhanced for Vehicle Tracking)
# Author: Alex Bewley (original)  |  Enhanced version for YOLO pipelines with ego-motion
# Features:
# - Safe empty match handling
# - Stable ID tracking (higher max_age, min_hits)
# - Velocity output from Kalman filter
# - Bounding-box clamping (never out of frame)
# - Confidence filtering (optional)

from __future__ import print_function
import numpy as np
from filterpy.kalman import KalmanFilter


# -------------------------- Utility Functions --------------------------

def linear_assignment(cost_matrix):
    """Solve linear assignment via LAP or SciPy Hungarian algorithm."""
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in range(len(y)) if y[i] >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def iou(bb_test, bb_gt):
    """Compute Intersection-over-Union (IoU) between two bboxes [x1,y1,x2,y2]."""
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1]) +
              (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh + 1e-6)
    return o


# -------------------------- Kalman Tracker --------------------------

class KalmanBoxTracker:
    """Represents the internal state of individual tracked objects (bboxes)."""
    count = 0

    def __init__(self, bbox, conf=1.0, cls_id=-1):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        # Constant velocity model
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
        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4] = self.convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.conf = conf
        self.cls_id = cls_id

    def update(self, bbox):
        """Update tracker with new detected bounding box."""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self.convert_bbox_to_z(bbox))

    def predict(self):
        """Advance state vector and return predicted bbox."""
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self, frame_shape=None):
        """Return current bbox estimate, optionally clamped to frame size."""
        bbox = self.convert_x_to_bbox(self.kf.x)
        if frame_shape is not None:
            h, w = frame_shape[:2]
            bbox[0, 0] = np.clip(bbox[0, 0], 0, w-1)
            bbox[0, 1] = np.clip(bbox[0, 1], 0, h-1)
            bbox[0, 2] = np.clip(bbox[0, 2], 0, w-1)
            bbox[0, 3] = np.clip(bbox[0, 3], 0, h-1)
        return bbox

    def get_velocity(self):
        """Return estimated velocity (vx, vy) from Kalman state."""
        vx = float(self.kf.x[4])
        vy = float(self.kf.x[5])
        return vx, vy

    @staticmethod
    def convert_bbox_to_z(bbox):
        """Convert [x1,y1,x2,y2] to [x,y,s,r] for Kalman filter."""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.
        y = bbox[1] + h / 2.
        s = w * h
        r = w / float(h)
        return np.array([x, y, s, r]).reshape((4, 1))

    @staticmethod
    def convert_x_to_bbox(x, score=None):
        """Convert [x,y,s,r] back to [x1,y1,x2,y2]."""
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        if score is None:
            return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))
        else:
            return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1, 5))


# -------------------------- SORT Multi-Object Tracker --------------------------

class Sort:
    def __init__(self, max_age=30, min_hits=5, iou_threshold=0.3, conf_thresh=0.2):
        """
        max_age: maximum number of frames to keep a track alive without detections
        min_hits: minimum number of hits before track is initialized
        iou_threshold: minimum IoU for valid match
        conf_thresh: discard detections below this confidence
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.conf_thresh = conf_thresh
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5)), frame_shape=None):
        """
        Params:
          dets - array of detections [x1,y1,x2,y2,score]
          frame_shape - optional (H,W) for clamping
        Returns:
          array of tracks [x1,y1,x2,y2,id]
        """
        self.frame_count += 1

        # Filter detections by confidence
        if len(dets) > 0:
            dets = dets[dets[:, 4] > self.conf_thresh]
        else:
            dets = np.empty((0, 5))

        # Predict next positions from existing trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []

        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(dets, trks)

        # Update matched trackers
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :4])

        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :4], conf=float(dets[i, 4]))
            self.trackers.append(trk)

        # Prepare output
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state(frame_shape)[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

    # -------------------------- Matching --------------------------

    def associate_detections_to_trackers(self, detections, trackers):
        """Match detections to existing trackers using IoU and Hungarian algorithm."""
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0), dtype=int)

        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = iou(det, trk)

        matched_indices = linear_assignment(-iou_matrix)

        # Safe empty handling
        if matched_indices.size == 0:
            matched_indices = np.empty((0, 2), dtype=int)

        unmatched_detections = [d for d in range(len(detections)) if d not in matched_indices[:, 0]]
        unmatched_trackers = [t for t in range(len(trackers)) if t not in matched_indices[:, 1]]

        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))

        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
