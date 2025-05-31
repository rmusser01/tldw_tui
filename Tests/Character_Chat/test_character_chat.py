# tests/test_character_chat_lib_property.py
import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
import re

from Character_Chat_Lib import (
    replace_placeholders,
    extract_character_id_from_ui_choice,
    parse_v1_card, parse_v2_card,
    validate_v2_card,
    process_db_messages_to_ui_history
)


# --- Property tests for replace_placeholders ---
@given(st.text(), st.text() | st.none(), st.text() | st.none())
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_property_replace_placeholders_no_char_placeholder_if_char_name_given(text, char_name, user_name):
    result = replace_placeholders(text, char_name, user_name)
    if char_name is not None:
        assert "{{char}}" not in result
        assert "<CHAR>" not in result
    if user_name is not None:
        assert "{{user}}" not in result
        assert "{{random_user}}" not in result
        assert "<USER>" not in result


@given(st.one_of(st.none(), st.just("")), st.text() | st.none(), st.text() | st.none())
def test_property_replace_placeholders_empty_or_none_text(text, char_name, user_name):
    assert replace_placeholders(text, char_name, user_name) == ""


# --- Property tests for extract_character_id_from_ui_choice ---
@given(st.text(min_size=1, max_size=50), st.integers(min_value=0, max_value=10000))
def test_property_extract_id_format1(name_part, char_id):
    name_part_cleaned = re.sub(r'[\(\)]', '', name_part)  # Avoid nested parens interfering
    if not name_part_cleaned.strip(): name_part_cleaned = "Char"  # Ensure name_part isn't just parens

    choice = f"{name_part_cleaned} (ID: {char_id})"
    assert extract_character_id_from_ui_choice(choice) == char_id


@given(st.integers(min_value=0, max_value=10000))
def test_property_extract_id_format2(char_id):
    choice = str(char_id)
    assert extract_character_id_from_ui_choice(choice) == char_id


@given(st.text().filter(lambda x: not re.fullmatch(r"\d+", x) and not re.search(r'\(ID:\s*\d+\s*\)$', x)))
@settings(suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
def test_property_extract_id_invalid_format_raises_valueerror(invalid_choice_text):
    # Ensure it's not an empty string which is a separate path in the function
    if not invalid_choice_text:
        invalid_choice_text = "abc"  # substitute if empty string generated
    with pytest.raises(ValueError):
        extract_character_id_from_ui_choice(invalid_choice_text)


# --- Property tests for V1/V2 card parsing (basic properties) ---
# Define strategies for V1 and V2 card structures
# This can get complex. For simplicity, let's test a core property.

# Minimal strategy for a valid V1 card structure fragment
v1_core_fields_strat = st.fixed_dictionaries({
    "name": st.text(min_size=1),
    "description": st.text(),
    "personality": st.text(),
    "scenario": st.text(),
    "first_mes": st.text(),
    "mes_example": st.text(),
})


@given(data=v1_core_fields_strat)
def test_property_parse_v1_card_preserves_name_and_maps_first_mes(data):
    parsed = parse_v1_card(data)
    assert parsed is not None
    assert parsed['name'] == data['name']
    assert parsed['first_message'] == data['first_mes']


# Minimal strategy for a valid V2 card structure fragment
v2_data_node_strat = st.fixed_dictionaries({
    "name": st.text(min_size=1),
    "description": st.text(),
    "personality": st.text(),
    "scenario": st.text(),
    "first_mes": st.text(),
    "mes_example": st.text(),
})
v2_card_strat = st.fixed_dictionaries({
    "spec": st.just("chara_card_v2"),
    "spec_version": st.just("2.0"),
    "data": v2_data_node_strat
})


@given(data=v2_card_strat)
def test_property_parse_v2_card_preserves_name_and_maps_first_mes(data):
    parsed = parse_v2_card(data)
    assert parsed is not None
    assert parsed['name'] == data['data']['name']
    assert parsed['first_message'] == data['data']['first_mes']


# --- Property tests for process_db_messages_to_ui_history ---
db_message_strat = st.fixed_dictionaries({
    "sender": st.sampled_from(["User", "TestChar", "OtherSender"]),
    "content": st.text(max_size=100),
    # "image_data": st.none() | st.binary(max_size=10), # Optional
    # "image_mime_type": st.none() | st.just("image/png"), # Optional
    # "timestamp": st.datetimes() # Optional for this func's logic
})
db_messages_list_strat = st.lists(db_message_strat, max_size=10)


@given(
    db_messages=db_messages_list_strat,
    char_name=st.text(min_size=1, max_size=20),
    user_name=st.text(min_size=1, max_size=20) | st.none()
)
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_property_process_db_messages_to_ui_history_output_length(db_messages, char_name, user_name):
    if any(msg['sender'] == "TestChar" for msg in
           db_messages) and char_name == "TestChar":  # ensure char_name used matches a sender
        ui_history = process_db_messages_to_ui_history(db_messages, char_name, user_name, "User", "TestChar")

        # Number of tuples in UI history should be related to messages, but not strictly equal due to pairing.
        # Max tuples is len(db_messages). Min is ceil(len(db_messages)/2) if perfect pairing.
        assert len(ui_history) <= len(db_messages)

        total_messages_in_output = 0
        for user_msg, bot_msg in ui_history:
            if user_msg is not None: total_messages_in_output += 1
            if bot_msg is not None: total_messages_in_output += 1

        # Every original message should appear in the output, unless it's empty or None (which our strat avoids for content)
        # This property is hard to state perfectly without reconstructing the logic.
        # A simpler one: if all messages are from User, all bot_msg in output are None.
        if all(m['sender'] == "User" for m in db_messages):
            assert all(pair[1] is None for pair in ui_history)

        # If all messages are from TestChar, all user_msg in output are None.
        elif all(m['sender'] == "TestChar" for m in db_messages):
            assert all(pair[0] is None for pair in ui_history)

# TODO: Add more property tests for:
# - `validate_v2_card`, `validate_character_book`, `validate_character_book_entry`
#   (generate structures, check that valid structures pass, introduce errors and check they fail with specific messages)
# - Other parsing functions where input variability is high.
# - Functions with complex branching logic based on input values.