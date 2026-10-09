"""COCO 17-keypoint skeleton overlay, independent of event severity."""
import math

EDGES = ((0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
         (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16))


def valid_point(point, threshold):
    return (len(point) >= 3 and all(math.isfinite(float(v)) for v in point[:3])
            and point[2] >= threshold and point[0] > 0 and point[1] > 0)


def draw_skeletons(image, people, threshold=.5):
    import cv2
    count = 0
    for points in people:
        if len(points) != 17:
            continue
        visible = [valid_point(p, threshold) for p in points]
        if not any(visible):
            continue
        count += 1
        for a,b in EDGES:
            if visible[a] and visible[b]:
                cv2.line(image, tuple(map(int, points[a][:2])), tuple(map(int, points[b][:2])),
                         (240,210,40), 2, cv2.LINE_AA)
        for p, ok in zip(points, visible):
            if ok:
                cv2.circle(image, tuple(map(int,p[:2])), 3, (90,250,160), -1)
    return count
