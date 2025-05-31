# test_property_character_chat_lib.py

import unittest
from hypothesis import given, strategies as st, settings, HealthCheck
import re

import Character_Chat_Lib as ccl

class TestCharacterChatLibProperty(unittest.TestCase):

    # --- replace_placeholders ---
    @given(text=st.text(),
           char_name=st.one_of(st.none(), st.text(min_size=1, max_size=50)),
           user_name=st.one_of(st.none(), st.text(min_size=1, max_size=50)))
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_replace_placeholders_properties(self, text, char_name, user_name):
        processed = ccl.replace_placeholders(text, char_name, user_name)
        self.assertIsInstance(processed, str)

        expected_char = char_name if char_name else "Character"
        expected_user = user_name if user_name else "User"

        if "{{char}}" in text:
            self.assertIn(expected_char, processed)
        if "{{user}}" in text:
            self.assertIn(expected_user, processed)
        if "<CHAR>" in text:
            self.assertIn(expected_char, processed)
        if "<USER>" in text:
            self.assertIn(expected_user, processed)

        # If no placeholders, text should be identical
        placeholders = ['{{char}}', '{{user}}', '{{random_user}}', '<USER>', '<CHAR>']
        if not any(p in text for p in placeholders):
            self.assertEqual(processed, text)

    @given(text=st.one_of(st.none(), st.just("")), char_name=st.text(), user_name=st.text())
    def test_replace_placeholders_empty_or_none_input_text(self, text, char_name, user_name):
        self.assertEqual(ccl.replace_placeholders(text, char_name, user_name), "")

    # --- replace_user_placeholder ---
    @given(history=st.lists(st.tuples(st.one_of(st.none(), st.text()), st.one_of(st.none(), st.text()))),
           user_name=st.one_of(st.none(), st.text(min_size=1, max_size=50)))
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_replace_user_placeholder_properties(self, history, user_name):
        processed_history = ccl.replace_user_placeholder(history, user_name)
        self.assertEqual(len(processed_history), len(history))
        expected_user = user_name if user_name else "User"

        for i, (original_user_msg, original_bot_msg) in enumerate(history):
            processed_user_msg, processed_bot_msg = processed_history[i]
            if original_user_msg is not None:
                self.assertIsInstance(processed_user_msg, str)
                if "{{user}}" in original_user_msg:
                    self.assertIn(expected_user, processed_user_msg)
                else:
                    self.assertEqual(processed_user_msg, original_user_msg)
            else:
                self.assertIsNone(processed_user_msg)

            if original_bot_msg is not None:
                self.assertIsInstance(processed_bot_msg, str)
                if "{{user}}" in original_bot_msg:
                    self.assertIn(expected_user, processed_bot_msg)
                else:
                    self.assertEqual(processed_bot_msg, original_bot_msg)
            else:
                self.assertIsNone(processed_bot_msg)

    # --- extract_character_id_from_ui_choice ---
    @given(name=st.text(alphabet=st.characters(min_codepoint=65, max_codepoint=122), min_size=1, max_size=20).filter(lambda x: '(' not in x and ')' not in x),
           id_val=st.integers(min_value=0, max_value=10**9))
    def test_extract_id_format_name_id(self, name, id_val):
        choice = f"{name} (ID: {id_val})"
        self.assertEqual(ccl.extract_character_id_from_ui_choice(choice), id_val)

    @given(id_val=st.integers(min_value=0, max_value=10**9))
    def test_extract_id_format_just_id(self, id_val):
        choice = str(id_val)
        self.assertEqual(ccl.extract_character_id_from_ui_choice(choice), id_val)

    @given(text=st.text().filter(lambda x: not re.search(r'\(\s*ID\s*:\s*\d+\s*\)\s*$', x) and not x.isdigit() and x != ""))
    def test_extract_id_invalid_format_raises_valueerror(self, text):
        with self.assertRaises(ValueError):
            ccl.extract_character_id_from_ui_choice(text)

    @given(choice=st.just(""))
    def test_extract_id_empty_string_raises_valueerror(self, choice):
         with self.assertRaises(ValueError):
            ccl.extract_character_id_from_ui_choice(choice)

    # --- process_db_messages_to_ui_history ---
    # This one is complex for property-based testing due to stateful accumulation.
    # We can test some basic properties.
    @given(db_messages=st.lists(st.fixed_dictionaries({
                'sender': st.sampled_from(["User", "TestChar", "OtherSender"]),
                'content': st.text(max_size=100)
           }), max_size=10),
           char_name=st.text(min_size=1, max_size=20),
           user_name=st.one_of(st.none(), st.text(min_size=1, max_size=20)))
    @settings(suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
    def test_process_db_messages_to_ui_history_output_structure(self, db_messages, char_name, user_name):
        if not db_messages: # Avoid issues with empty messages list if logic depends on non-empty
            return

        ui_history = ccl.process_db_messages_to_ui_history(db_messages, char_name, user_name,
                                                           actual_char_sender_id_in_db=char_name) # Map TestChar to char_name
        self.assertIsInstance(ui_history, list)
        for item in ui_history:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)
            self.assertTrue(item[0] is None or isinstance(item[0], str))
            self.assertTrue(item[1] is None or isinstance(item[1], str))

        # If all messages are from User, bot messages in UI should be None
        if all(msg['sender'] == "User" for msg in db_messages):
            for _, bot_msg in ui_history:
                self.assertIsNone(bot_msg)

        # If all messages are from Character, user messages in UI should be None
        if all(msg['sender'] == char_name for msg in db_messages):
            for user_msg, _ in ui_history:
                self.assertIsNone(user_msg)

    # --- Card Validation Properties (Example for validate_character_book_entry) ---
    # Strategy for a valid character book entry core
    valid_entry_core_st = st.fixed_dictionaries({
        'keys': st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=5),
        'content': st.text(min_size=1, max_size=200),
        'enabled': st.booleans(),
        'insertion_order': st.integers()
    })

    @given(entry_core=valid_entry_core_st,
           entry_id_set=st.sets(st.integers(min_value=0, max_value=1000)))
    def test_validate_character_book_entry_valid_core(self, entry_core, entry_id_set):
        is_valid, errors = ccl.validate_character_book_entry(entry_core, 0, entry_id_set)
        self.assertTrue(is_valid, f"Errors for supposedly valid core: {errors}")
        self.assertEqual(len(errors), 0)

    @given(entry_core=valid_entry_core_st,
           bad_key_type=st.integers(), # Make keys not a list of strings
           entry_id_set=st.sets(st.integers()))
    def test_validate_character_book_entry_invalid_keys_type(self, entry_core, bad_key_type, entry_id_set):
        invalid_entry = {**entry_core, 'keys': bad_key_type}
        is_valid, errors = ccl.validate_character_book_entry(invalid_entry, 0, entry_id_set)
        self.assertFalse(is_valid)
        self.assertTrue(any("Field 'keys' must be of type 'list'" in e for e in errors))

    # More properties can be added for other parsing/validation functions if they
    # have clear invariants that can be tested with generated data.


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)