from wetlandbirds.data.splits import make_split_map


def test_split_map():
    result = make_split_map({"train": ["a.mp4"], "test": ["b.mp4"]})
    assert result["a.mp4"] == "train"
    assert result["b.mp4"] == "test"
