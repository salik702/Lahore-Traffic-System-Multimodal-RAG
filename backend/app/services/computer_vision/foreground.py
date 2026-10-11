"""Select a likely foreground subject without claiming calibrated camera distance."""
from .rider_association import area


def foreground_vehicle(vehicles, image_size):
    candidates = [v for v in vehicles if v.get('class', v.get('vehicle_type')) != 'bicycle'] or vehicles
    if not candidates:
        return None
    width = max(1, image_size.get('width', 1))
    height = max(1, image_size.get('height', 1))

    def score(vehicle):
        fraction = min(1, area(vehicle['bbox']) / (width * height))
        bottom = max(0, min(1, vehicle['bbox'][3] / height))
        return fraction ** .5 * (.5 + .5 * bottom)

    return max(candidates, key=score)


def subject_bounds(vehicle, riders, image_shape):
    """Include uniquely supported riders' heads, not only the motorcycle chassis."""
    regions = [vehicle['bbox']] + [r['person_bbox'] for r in riders
        if r['vehicle_id'] == vehicle['id'] and r['association'] == 'supported']
    height, width = image_shape[:2]
    x1, y1 = min(b[0] for b in regions), min(b[1] for b in regions)
    x2, y2 = max(b[2] for b in regions), max(b[3] for b in regions)
    padding = min(24, max(8, (x2 - x1) * .025))
    return [max(0, int(x1 - padding)), max(0, int(y1 - padding)),
            min(width, int(x2 + padding)), min(height, int(y2 + padding))]


def report_scope(result, vehicle_id=None):
    """Only server-owned observations for one vehicle enter its report and totals."""
    vehicles = result['vehicle_results']
    target = vehicle_id or result.get('primary_vehicle_id')
    if target is None:
        chosen = foreground_vehicle(vehicles, result.get('image_size', {}))
        target = chosen['vehicle_id'] if chosen else None
    selected = next((v for v in vehicles if v['vehicle_id'] == target), None)
    if target is not None and selected is None:
        raise ValueError('Requested report vehicle does not belong to this analysis')
    from .fines import fine_summary
    scoped = dict(result)
    scoped['vehicle_results'] = [selected] if selected else []
    scoped['potential_violations'] = selected['violations'] if selected else []
    scoped['summary'] = selected['summary'] if selected else fine_summary([], [])
    scoped['sources'] = list({s['url']: s for v in scoped['potential_violations'] for s in v['sources']}.values())
    scoped['evidence_crops'] = [c for c in result.get('evidence_crops', []) if selected and c['vehicle_id'] == target]
    scoped['report_vehicle_id'] = target
    scoped['report_scope'] = 'One selected vehicle only; surrounding vehicles and their penalties are excluded.'
    if selected:
        scoped['original_image_url'] = selected.get('image_url') or result['original_image_url']
        scoped['annotated_image_url'] = selected.get('focus_annotated_image_url') or scoped['original_image_url']
    return scoped
