import numpy as np
from scipy.optimize import linear_sum_assignment

def iou_batch(bb_test, bb_gt):
    """
    Computes IoU between two sets of bounding boxes.
    bb_test: array of shape (N, 4) where boxes are [x1, y1, x2, y2]
    bb_gt: array of shape (M, 4)
    Returns: matrix of shape (N, M) with IoU values
    """
    bb_test = np.expand_dims(bb_test, 1) # N x 1 x 4
    bb_gt = np.expand_dims(bb_gt, 0)     # 1 x M x 4

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])

    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)

    wh = w * h
    denominator = ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1]) 
                   + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    # avoid division by zero
    denominator = np.where(denominator <= 0, 1e-5, denominator)
    o = wh / denominator
    return o

def bbox_to_z(bbox):
    """
    Converts bbox [x1, y1, x2, y2] to measurement format [x, y, s, r]
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h
    r = w / float(h) if h > 0 else 0
    return np.array([x, y, s, r]).reshape((4, 1))

def x_to_bbox(x):
    """
    Converts state vector [x, y, s, r, vx, vy, vs] to bbox [x1, y1, x2, y2]
    """
    cx, cy, s, r = x[0, 0], x[1, 0], x[2, 0], x[3, 0]
    if s < 0:
        s = 0
    if r <= 0:
        r = 1e-5
    w = np.sqrt(s * r)
    h = s / w if w > 0 else 0
    x1 = cx - w / 2.
    y1 = cy - h / 2.
    x2 = cx + w / 2.
    y2 = cy + h / 2.
    return np.array([x1, y1, x2, y2])

class KalmanBoxTracker:
    count = 0
    def __init__(self, bbox, cls_name="car"):
        """
        Initializes a tracker using initial bounding box.
        """
        # State vector: [x, y, s, r, vx, vy, vs]
        self.x = np.zeros((7, 1))
        self.x[0:4] = bbox_to_z(bbox)
        
        # State covariance
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 1000.0, 1000.0, 1000.0])
        
        # Process noise covariance
        self.Q = np.diag([1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001])
        
        # Measurement noise covariance
        self.R = np.diag([1.0, 1.0, 10.0, 10.0])
        
        # Transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix
        self.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0], # wait, H measurement is just state [x, y, s, r] so it should be:
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        # Wait, the second row has a 1 in column 5! Let's fix that measurement matrix.
        self.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 1 # confirmation counter
        self.cls_name = cls_name

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        
        z = bbox_to_z(bbox)
        
        # Kalman update
        y = z - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(self.P, np.dot(self.H.T, np.linalg.inv(S)))
        self.x = self.x + np.dot(K, y)
        self.P = np.dot(np.eye(7) - np.dot(K, self.H), self.P)

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if (self.x[2, 0] + self.x[6, 0]) <= 0:
            self.x[6, 0] *= 0.0
            
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        
        self.time_since_update += 1
        self.history.append(x_to_bbox(self.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return x_to_bbox(self.x)

class Sort:
    def __init__(self, max_age=10, min_hits=3, iou_threshold=0.30):
        """
        Sets key parameters for SORT.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        KalmanBoxTracker.count = 0

    def update(self, dets, cls_names):
        """
        dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],...]
        cls_names - list of class names corresponding to each detection
        
        Returns a list of active confirmed trackers: (bbox, id, cls_name)
        """
        # 1. Predict state for all existing trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            trks[t, :] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
                
        self.trackers = [self.trackers[i] for i in range(len(self.trackers)) if i not in to_del]
        trks = np.delete(trks, to_del, axis=0)

        # 2. Match detections with trackers
        if len(dets) > 0 and len(trks) > 0:
            iou_matrix = iou_batch(dets[:, :4], trks[:, :4])
            
            if min(iou_matrix.shape) > 0:
                row_ind, col_ind = linear_sum_assignment(-iou_matrix)
                matched_indices = np.stack((row_ind, col_ind), axis=1)
            else:
                matched_indices = np.empty((0, 2), dtype=int)
        else:
            matched_indices = np.empty((0, 2), dtype=int)

        # 3. Filter matched indices based on IoU threshold
        unmatched_detections = []
        for d, det in enumerate(dets):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)
                
        unmatched_trackers = []
        for t, trk in enumerate(self.trackers):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)

        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m)

        # 4. Update matched trackers
        for m in matches:
            self.trackers[m[1]].update(dets[m[0], :4])
            self.trackers[m[1]].cls_name = cls_names[m[0]]

        # 5. Create new trackers for unmatched detections
        for i in unmatched_detections:
            trk = KalmanBoxTracker(dets[i, :4], cls_names[i])
            self.trackers.append(trk)

        # 6. Retrieve active confirmed trackers and prune expired ones
        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
                continue
                
            if trk.hits >= self.min_hits:
                ret.append((d, trk.id, trk.cls_name))
                
        return ret
