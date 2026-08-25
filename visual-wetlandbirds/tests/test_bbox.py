from wetlandbirds.data.bbox import parse_bbox_cell


def test_parse_list_of_boxes():
    boxes = parse_bbox_cell("[(1, 2, 10, 20), (5, 6, 15, 25)]")
    assert len(boxes) == 2
    assert boxes[0]["box"] == (1.0, 2.0, 10.0, 20.0)


def test_parse_empty():
    assert parse_bbox_cell("") == []
