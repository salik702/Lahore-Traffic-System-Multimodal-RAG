"""Recover foreground leg detail lost in a full-scene pose resize, within a budget."""
from .detector import predict
from .policy import policy
from .quality import crop_region, assess_quality
from .rider_association import area, iou, associate_riders


def refine_riders(image, detections, poses):
    initial = associate_riders(detections, poses)
    uncertain = sorted((r for r in initial if r['association'] == 'uncertain'),
                       key=lambda r: area(r['person_bbox']), reverse=True)
    refined = list(poses)
    for rider in uncertain[:policy()['max_pose_refinements']]:
        p = rider['person_bbox']
        bounds = [max(0, int(p[0])-24), max(0, int(p[1])-24),
                  min(image.shape[1], int(p[2])+24), min(image.shape[0], int(p[3])+24)]
        crop = crop_region(image, bounds)
        if not assess_quality(crop)['sufficient']:
            continue
        boxes, result = predict('pose', crop, .4, 640)
        if result.keypoints is None:
            continue
        local = []
        for box, points in zip(boxes, result.keypoints.data.tolist()):
            b = box['bbox']
            local.append({'bbox': [b[0]+bounds[0], b[1]+bounds[1], b[2]+bounds[0], b[3]+bounds[1]],
                          'points': [[q[0]+bounds[0], q[1]+bounds[1], q[2]] for q in points],
                          'person_id': rider['person_id'], 'source': 'person_crop'})
        best = max(local, key=lambda q: iou(q['bbox'], p), default=None)
        if best and iou(best['bbox'], p) > .6:
            refined.append(best)
    return associate_riders(detections, refined)
