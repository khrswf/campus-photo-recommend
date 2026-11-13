"""Filter reference photos using catalog labels."""


def filter_photos(
    photos, *, location_id=None, person_count=None, framing=None, pose=None,
    forbidden_poses=(), required_objects=(),
):
    """Return confirmed photos, pending checks, and exclusions with reasons."""
    matches = []
    pending = []
    excluded = []
    for photo in photos:
        mismatches = []
        unknowns = []
        for field, expected in (
            ("location_id", location_id),
            ("person_count", person_count),
            ("framing", framing),
            ("pose", pose),
        ):
            if expected is None:
                continue
            actual = photo.get(field)
            if actual is None:
                unknowns.append(f"{field}_unknown")
            elif actual != expected:
                mismatches.append(f"{field}_mismatch")

        if forbidden_poses:
            actual_pose = photo.get("pose")
            if actual_pose is None:
                if "pose_unknown" not in unknowns:
                    unknowns.append("pose_unknown")
            elif actual_pose in forbidden_poses:
                mismatches.append("forbidden_pose")

        if required_objects:
            objects = photo.get("background_objects")
            if objects is None:
                unknowns.append("background_objects_unknown")
            elif not all(item in objects for item in required_objects):
                mismatches.append("required_objects_missing")

        if photo.get("viewpoint_status") == "unavailable":
            mismatches.append("viewpoint_unavailable")

        if mismatches:
            excluded.append({"photo_id": photo["photo_id"], "reasons": mismatches})
        elif unknowns:
            pending.append({"photo_id": photo["photo_id"], "reasons": unknowns})
        else:
            matches.append(photo)
    return matches, pending, excluded
