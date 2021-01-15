from deepspeech2.data import TextTransform


def test_text_transform():
    text_transform = TextTransform()
    assert len(text_transform.char_map) == 28
    assert text_transform.text_to_int("HELLO".lower()) == [9, 6, 13, 13, 16]
    assert text_transform.text_to_int("HI THERE".lower()) == [9, 10, 1, 21, 9, 6, 19, 6]
    assert text_transform.int_to_text([9, 6, 13, 13, 16]) == 'hello'
