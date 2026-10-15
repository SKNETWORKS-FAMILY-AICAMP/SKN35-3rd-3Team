from pathlib import Path
from streamlit.testing.v1 import AppTest
from streamlit.proto.WidgetStates_pb2 import WidgetStates

app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "main.py")).run(timeout=30)
next(button for button in app.button if button.key == "nav_이용 안내").click().run(timeout=30)
assert not app.exception, str(app.exception)
assert len(app.expander) == 4
cards = [element for element in app if "-law_guide_step_" in getattr(getattr(element, "proto", None), "id", "")]
assert len(cards) == 4
for card in cards:
    assert card.proto.flex_container.border
    assert len(card.markdown) == 1
    assert 'class="law-step-layout"' in card.markdown[0].value
    assert 'class="law-step-number"' in card.markdown[0].value
    assert 'class="law-step-title"' in card.markdown[0].value
    assert 'class="law-step-description"' in card.markdown[0].value
assert not any("law-guide-step-card" in element.proto.body for element in app.get("html"))

def toggle(index, expanded):
    # AppTest lacks expander.click(); send the same boolean widget event
    # that the browser emits, exercising Streamlit's registered callbacks.
    widgets = WidgetStates()
    for current, expander in enumerate(app.expander):
        widget = widgets.widgets.add()
        widget.id = expander.proto.id
        widget.bool_value = expanded if current == index else expander.proto.expanded
    app._run(widgets, timeout=30)
    assert not app.exception, str(app.exception)
    return [expander.proto.expanded for expander in app.expander]

assert toggle(0, True) == [True, False, False, False]
assert toggle(1, True) == [False, True, False, False]
assert toggle(3, True) == [False, False, False, True]
assert toggle(3, False) == [False, False, False, False]
assert toggle(0, True) == [True, False, False, False]
print("FAQ exclusive open, switching, close and reopen OK")
