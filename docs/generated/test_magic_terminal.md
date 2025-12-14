## AI Summary

A file named test_magic_terminal.py.


### Function: check_cpaste(code, should_fail)

**Description:** Execute code via 'cpaste' and ensure it was executed, unless
should_fail is set.

### Function: test_cpaste()

**Description:** Test cpaste magic

## Class: PasteTestCase

**Description:** Multiple tests for clipboard pasting

### Function: runf()

**Description:** Marker function: sets a flag when executed.
        

### Function: paste(self, txt, flags)

**Description:** Paste input text, by default in quiet mode

### Function: setUp(self)

### Function: tearDown(self)

### Function: test_paste(self)

### Function: test_paste_pyprompt(self)

### Function: test_paste_py_multi(self)

### Function: test_paste_py_multi_r(self)

**Description:** Now, test that self.paste -r works

### Function: test_paste_email(self)

**Description:** Test pasting of email-quoted contents

### Function: test_paste_email2(self)

**Description:** Email again; some programs add a space also at each quoting level

### Function: test_paste_email_py(self)

**Description:** Email quoting of interactive input

### Function: test_paste_echo(self)

**Description:** Also test self.paste echoing, by temporarily faking the writer

### Function: test_paste_leading_commas(self)

**Description:** Test multiline strings with leading commas

### Function: test_paste_trailing_question(self)

**Description:** Test pasting sources with trailing question marks
