
import pytest
from src.classifier import load_classifier, predict_style
from src.config import CLASS_NAMES

@pytest.fixture(scope="module")
def model():
    return load_classifier()


def test_model_loads_without_error(model):
    assert model is not None


def test_prediction_returns_a_dictionary(model):
    result = predict_style(model, "test_image.jpg")

    assert isinstance(result, dict)
    assert "style" in result
    assert "confidence" in result
    assert "top_3" in result


def test_predicted_style_is_one_of_the_eight_trained_classes(model):
    result = predict_style(model, "test_image.jpg")

    assert result["style"] in CLASS_NAMES


def test_confidence_is_a_valid_probability(model):
   
    result = predict_style(model, "test_image.jpg")

    assert 0.0 <= result["confidence"] <= 1.0


def test_top_3_has_exactly_three_entries(model):
    result = predict_style(model, "test_image.jpg")

    assert len(result["top_3"]) == 3


def test_top_3_is_sorted_highest_confidence_first(model):
    result = predict_style(model, "test_image.jpg")

    confidences = [conf for style, conf in result["top_3"]]

    assert confidences[0] >= confidences[1] >= confidences[2]


def test_top_prediction_matches_first_entry_in_top_3(model):
    result = predict_style(model, "test_image.jpg")

    top_3_best_style = result["top_3"][0][0]

    assert result["style"] == top_3_best_style


def test_all_top_3_styles_are_valid_class_names(model):
    result = predict_style(model, "test_image.jpg")

    for style_name, confidence in result["top_3"]:
        assert style_name in CLASS_NAMES